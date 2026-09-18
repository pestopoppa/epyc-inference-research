"""Bounded, observation-only whole-lifecycle serving sampler.

One reader thread owns all proc/sys I/O. Launcher methods only publish markers and
wait for bounded acknowledgements; shutdown never waits on an I/O-held state lock.
The module records facts and unknowns, never a protocol verdict.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import types
from typing import Any


CONTEXT_SCHEMA = "epyc.autokernel.lifecycle_observation_context.v1"
SAMPLE_SCHEMA = "epyc.autokernel.lifecycle_observation_sample.v1"
OBSERVATION_SCHEMA = "epyc.autokernel.lifecycle_observation.v1"
INSTRUMENT_SCHEMA = "epyc.autokernel.loaded_serving_instrument.v1"
ARTIFACT_SCHEMA = "epyc.autokernel.prepared_artifact_identity.v1"
DETECTOR_VERSION = "autokernel-lifecycle-observer-2"
PHASES = ("setup", "load", "placement", "health", "warmup", "measurement",
          "teardown")
RUNTIME_WITNESS_STATES = ("unsupported", "compiled_unproven", "fired_under_target",
                          "unknown")
FOREIGN_KINDS = ("ordinary", "model_inference")
BUDGET_FIELDS = frozenset({
    "max_samples", "max_pending_markers", "max_processes", "max_read_bytes",
    "max_proc_entries", "max_retained_bytes", "max_map_entries", "max_fd_entries",
    "max_cpu_ids", "max_numa_rows",
    "max_dso_entries", "phase_ack_timeout_s", "join_timeout_s",
    "max_probe_duration_s",
})
CONTAINER_FIELDS = frozenset({"path", "dev", "ino", "uid", "nlink", "mode"})
WORKER_FIELDS = frozenset({
    "worker_id", "worker_incarnation", "grant_id", "grant_generation",
    "container_identity",
})


class ObservationError(RuntimeError):
    """An observation input, budget, identity, or captured state is invalid."""


class ObserverShutdownUnresolved(RuntimeError):
    """The containing owner must not start another unit while a reader remains live."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ObservationError("value is not finite JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _closed(value: Any, fields: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ObservationError(f"{label} has missing or unknown fields")
    return value


def _text(value: Any, label: str, *, maximum: int = 16 * 1024) -> str:
    if (not isinstance(value, str) or not value.strip() or "\0" in value
            or len(value.encode("utf-8")) > maximum):
        raise ObservationError(f"{label} must be bounded non-empty text")
    return value


def _sha(value: Any, label: str) -> str:
    value = _text(value, label, maximum=64)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ObservationError(f"{label} must be lowercase SHA-256")
    return value


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ObservationError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or (positive and value <= 0)):
        raise ObservationError(f"{label} must be finite" + (" and positive" if positive else ""))
    return float(value)


def required_sample_capacity(*, max_duration_s: float, cadence_s: float,
                             nonperiodic_samples: int) -> int:
    """Bound counts for a fixed marker schedule inside an enforced duration."""
    duration = _finite(max_duration_s, "observation maximum duration", positive=True)
    cadence = _finite(cadence_s, "observation cadence", positive=True)
    markers = _integer(nonperiodic_samples, "nonperiodic sample count")
    quotient = _finite(duration / cadence, "observation sample count ratio", positive=True)
    return math.ceil(quotient) + markers


def _utc(value: Any, label: str) -> str:
    value = _text(value, label)
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ObservationError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ObservationError(f"{label} must include timezone")
    return value


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical(value).decode("utf-8"))


def _bounded_entries(path: Path, maximum: int, label: str) -> list[Path]:
    rows: list[Path] = []
    try:
        with os.scandir(path) as stream:
            for entry in stream:
                if len(rows) >= maximum:
                    raise ObservationError(f"{label} entry budget exhausted at {maximum}")
                rows.append(Path(entry.path))
    except ObservationError:
        raise
    except OSError as exc:
        raise ObservationError(f"cannot enumerate {label}: {exc}") from exc
    return rows


def _bounded_bytes(path: Path, maximum: int, label: str) -> bytes:
    try:
        with path.open("rb") as stream:
            data = stream.read(maximum + 1)
    except OSError as exc:
        raise ObservationError(f"cannot read {label}: {exc}") from exc
    if len(data) > maximum:
        raise ObservationError(f"{label} byte budget exhausted at {maximum}")
    return data


def _bounded_text(path: Path, maximum: int, label: str) -> str:
    try:
        return _bounded_bytes(path, maximum, label).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ObservationError(f"{label} is not UTF-8") from exc


def parse_cpu_list(value: str, label: str = "CPU list", *, max_cpu_ids: int = 4096
                   ) -> frozenset[int]:
    value = _text(value, label).strip()
    maximum = _integer(max_cpu_ids, "max_cpu_ids", 1)
    result: set[int] = set()
    for part in value.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise ObservationError(f"{label} contains malformed range {part!r}")
        try:
            lo = int(match.group(1))
            hi = int(match.group(2) or lo)
        except ValueError as exc:
            raise ObservationError(f"{label} integer exceeds parser bound") from exc
        if lo > hi:
            raise ObservationError(f"{label} contains descending range {part!r}")
        if hi - lo + 1 > maximum or len(result) + hi - lo + 1 > maximum:
            raise ObservationError(f"{label} CPU expansion budget exhausted at {maximum}")
        result.update(range(lo, hi + 1))
    if not result:
        raise ObservationError(f"{label} is empty")
    if len(result) > maximum:
        raise ObservationError(f"{label} CPU expansion budget exhausted at {maximum}")
    return frozenset(result)


def parse_proc_stat(text: str, expected_pid: int) -> dict[str, Any]:
    opening, closing = text.find("("), text.rfind(")")
    if opening <= 0 or closing <= opening:
        raise ObservationError(f"malformed stat row for PID {expected_pid}")
    fields = text[closing + 1:].split()
    try:
        pid = int(text[:opening].strip())
        ticks = int(fields[11]) + int(fields[12])
        start_ticks = int(fields[19])
        processor = int(fields[36])
    except (IndexError, ValueError) as exc:
        raise ObservationError(f"malformed stat fields for PID {expected_pid}") from exc
    if pid != expected_pid or ticks < 0 or start_ticks <= 0 or processor < 0:
        raise ObservationError(f"invalid stat identity/counters for PID {expected_pid}")
    return {"pid": pid, "comm": text[opening + 1:closing], "cpu_ticks": ticks,
            "start_ticks": start_ticks, "processor": processor}


def _stat_identity(path: Path) -> dict[str, Any]:
    try:
        row = path.stat()
    except OSError as exc:
        raise ObservationError(f"cannot stat identity {path}: {exc}") from exc
    return {"path": str(path), "dev": row.st_dev, "ino": row.st_ino,
            "uid": row.st_uid, "nlink": row.st_nlink, "mode": row.st_mode}


def prepare_artifact_identity(path: Path, *, max_bytes: int = 512 * 1024 * 1024
                              ) -> dict[str, Any]:
    """Hash one immutable DSO before the observation window and pin its inode metadata."""
    path = Path(path)
    maximum = _integer(max_bytes, "artifact max_bytes", 1)
    try:
        before = path.stat()
    except OSError as exc:
        raise ObservationError(f"cannot stat prepared artifact {path}: {exc}") from exc
    if not path.is_file() or before.st_size > maximum:
        raise ObservationError("prepared artifact is not a bounded regular file")
    digest = hashlib.sha256(_bounded_bytes(path, maximum, "prepared artifact")).hexdigest()
    try:
        after = path.stat()
    except OSError as exc:
        raise ObservationError(f"cannot restat prepared artifact {path}: {exc}") from exc
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in fields):
        raise ObservationError("prepared artifact changed while it was hashed")
    body = {"schema": ARTIFACT_SCHEMA, "path": str(path), "sha256": digest,
            "size": after.st_size, "dev": after.st_dev, "ino": after.st_ino,
            "mtime_ns": after.st_mtime_ns, "ctime_ns": after.st_ctime_ns}
    return body | {"identity_sha256": _digest(body)}


def _validate_artifact(value: Any) -> dict[str, Any]:
    fields = {"schema", "path", "sha256", "size", "dev", "ino", "mtime_ns",
              "ctime_ns", "identity_sha256"}
    row = dict(_closed(value, fields, "prepared artifact identity"))
    if row["schema"] != ARTIFACT_SCHEMA:
        raise ObservationError("prepared artifact identity schema is unsupported")
    for key in ("size", "dev", "ino", "mtime_ns", "ctime_ns"):
        _integer(row[key], f"artifact {key}")
    _text(row["path"], "artifact path")
    _sha(row["sha256"], "artifact content digest")
    supplied = _sha(row.pop("identity_sha256"), "artifact identity digest")
    if supplied != _digest(row):
        raise ObservationError("prepared artifact identity digest mismatch")
    row["identity_sha256"] = supplied
    return row


def _stable_json_value(value: Any) -> tuple[str, Any]:
    if value is None or isinstance(value, (bool, int, str)):
        return "pinned", value
    if isinstance(value, float) and math.isfinite(value):
        return "pinned", value
    if isinstance(value, (tuple, list)):
        rows = [_stable_json_value(item) for item in value]
        if all(status == "pinned" for status, _ in rows):
            return "pinned", [item for _, item in rows]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        rows = {key: _stable_json_value(item) for key, item in value.items()}
        if all(status == "pinned" for status, _ in rows.values()):
            return "pinned", {key: item for key, (_, item) in rows.items()}
    return "unproven", None


def _code_projection(code: types.CodeType) -> tuple[str, dict[str, Any]]:
    constants, complete = [], True
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            status, value = _code_projection(item)
        elif (type(item) is frozenset and len(item) <= 4096
              and all(type(member) is str for member in item)):
            # CPython emits these for literal membership tests. Preserve the
            # existing outer type tag and sort only exact immutable strings.
            status, value = "pinned", sorted(item)
        elif type(item) is bytes and len(item) <= 4096:
            # Immutable code constants only. Configuration/object projection stays
            # unchanged; mutable buffers and bytes subclasses remain unproven.
            status, value = "pinned", item.hex()
        else:
            status, value = _stable_json_value(item)
        constants.append({"status": status, "value": value,
                          "type": f"{type(item).__module__}.{type(item).__qualname__}"})
        complete = complete and status == "pinned"
    body = {"bytecode": code.co_code.hex(), "constants": constants,
            "exception_table": code.co_exceptiontable.hex(),
            "names": list(code.co_names), "varnames": list(code.co_varnames),
            "freevars": list(code.co_freevars), "cellvars": list(code.co_cellvars),
            "argcount": code.co_argcount, "kwonlyargcount": code.co_kwonlyargcount,
            "posonlyargcount": code.co_posonlyargcount, "flags": code.co_flags,
            "nlocals": code.co_nlocals, "stacksize": code.co_stacksize}
    return ("pinned" if complete else "unproven"), body


def _bound_object_state(value: Any) -> tuple[str, Any]:
    """Project explicit instance state without treating ``__dict__`` as exhaustive."""
    state: dict[str, Any] = {}
    if hasattr(value, "__dict__"):
        try:
            state["dict"] = dict(vars(value))
        except (TypeError, ValueError):
            return "unproven", None
    slots: list[tuple[type[Any], str, str]] = []
    for owner in type(value).__mro__:
        owner_state = vars(owner)
        if owner is not object and "__slots__" not in owner_state \
                and "__dict__" not in owner_state:
            # A native base can carry behavior-bearing payload outside the explicit
            # Python instance dictionary/slot representations enumerated here.
            return "unproven", None
        declared = owner_state.get("__slots__", ())
        if isinstance(declared, str):
            declared = (declared,)
        if not isinstance(declared, (tuple, list)) \
                or not all(isinstance(name, str) for name in declared):
            return "unproven", None
        for name in declared:
            if name in {"__dict__", "__weakref__"}:
                continue
            attribute = (f"_{owner.__name__.lstrip('_')}{name}"
                         if name.startswith("__") and not name.endswith("__") else name)
            slots.append((owner, name, attribute))
    slot_state: dict[str, Any] = {}
    for owner, name, attribute in slots:
        key = f"{owner.__module__}.{owner.__qualname__}:{name}"
        descriptor = vars(owner).get(attribute)
        if descriptor is None or not hasattr(descriptor, "__get__"):
            return "unproven", None
        try:
            slot_state[key] = descriptor.__get__(value, type(value))
        except AttributeError:
            slot_state[key] = {"state": "unset"}
        except Exception:
            return "unproven", None
    if slots:
        state["slots"] = slot_state
    if not state:
        return _stable_json_value(value)
    return _stable_json_value(state)


def callable_identity(value: Callable[..., Any]) -> dict[str, Any]:
    """Describe loaded Python code and explicitly account for closure configuration."""
    if not callable(value):
        raise ObservationError("instrument callable is not callable")
    module = _text(getattr(value, "__module__", type(value).__module__), "callable module")
    qualname = _text(getattr(value, "__qualname__", type(value).__qualname__),
                     "callable qualname")
    code = getattr(value, "__code__", None)
    if code is None:
        return {"module": module, "qualname": qualname, "kind": "builtin_or_extension",
                "implementation_status": "unproven", "implementation_sha256": None,
                "configuration_status": "unproven",
                "configuration_sha256": None}
    implementation_status, code_body = _code_projection(code)
    bound_self = getattr(value, "__self__", None)
    if bound_self is None:
        bound_status, bound_state = "pinned", None
    else:
        bound_status, bound_state = _bound_object_state(bound_self)
    config = {"defaults": getattr(value, "__defaults__", None),
              "kwdefaults": getattr(value, "__kwdefaults__", None),
              "bound_self": bound_state, "closure": []}
    cells = getattr(value, "__closure__", None) or ()
    for cell in cells:
        try:
            config["closure"].append(cell.cell_contents)
        except ValueError:
            config["closure"].append("<empty-cell>")
    status, stable_config = _stable_json_value(config)
    if bound_status != "pinned":
        status, stable_config = "unproven", None
    return {"module": module, "qualname": qualname, "kind": "python",
            "implementation_status": implementation_status,
            "implementation_sha256": _digest(code_body),
            "configuration_status": status,
            "configuration_sha256": _digest(stable_config) if status == "pinned" else None}


def loaded_instrument_identity(*, measurement_callable: Callable[..., Any],
                               clock_callable: Callable[..., Any],
                               supporting_callables: Sequence[Callable[..., Any]],
                               used_constants: Mapping[str, Any],
                               dependency_packages: Sequence[str]) -> dict[str, Any]:
    """Build an honest identity for the loaded evaluator/timer implementation scope."""
    if not isinstance(supporting_callables, (list, tuple)):
        raise ObservationError("supporting_callables must be a sequence")
    if len(supporting_callables) > 64 or len(dependency_packages) > 64 \
            or len(used_constants) > 256:
        raise ObservationError("instrument identity input count exceeds bound")
    callables = [callable_identity(measurement_callable), callable_identity(clock_callable)]
    callables.extend(callable_identity(item) for item in supporting_callables)
    constants_status, constants = _stable_json_value(dict(used_constants))
    if constants_status != "pinned":
        raise ObservationError("used instrument constants are not stably serializable")
    dependencies: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in dependency_packages:
        package = _text(package, "dependency package", maximum=256)
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    body = {
        "schema": INSTRUMENT_SCHEMA,
        "scope": "loaded measurement, timing, named supporting callables and used constants",
        "identity_basis": "loaded Python code projection; extension code version-only",
        "measurement_callable": callables[0], "clock_callable": callables[1],
        "supporting_callables": callables[2:], "used_constants": constants,
        "dependency_versions": dependencies,
        "configuration_complete": all(dependencies.values()) and all(
            row["configuration_status"] == "pinned"
            and row["implementation_status"] == "pinned" for row in callables),
    }
    return body | {"sha256": _digest(body)}


def validate_instrument_identity(value: Any) -> dict[str, Any]:
    fields = {"schema", "scope", "identity_basis", "measurement_callable",
              "clock_callable", "supporting_callables", "used_constants",
              "dependency_versions", "configuration_complete", "sha256"}
    row = dict(_closed(value, fields, "loaded instrument identity"))
    if row["schema"] != INSTRUMENT_SCHEMA:
        raise ObservationError("loaded instrument identity schema is unsupported")
    callable_fields = {"module", "qualname", "kind", "implementation_status",
                       "implementation_sha256",
                       "configuration_status", "configuration_sha256"}
    callable_rows = [row["measurement_callable"], row["clock_callable"],
                     *row["supporting_callables"]]
    for item in callable_rows:
        item = _closed(item, callable_fields, "loaded callable identity")
        if item["kind"] not in {"python", "builtin_or_extension"}:
            raise ObservationError("loaded callable kind is unsupported")
        if item["implementation_status"] not in {"pinned", "unproven"}:
            raise ObservationError("callable implementation status is unsupported")
        if item["configuration_status"] not in {"pinned", "unproven"}:
            raise ObservationError("callable configuration status is unsupported")
        if item["implementation_status"] == "pinned":
            _sha(item["implementation_sha256"], "callable implementation digest")
        if item["configuration_status"] == "pinned":
            _sha(item["configuration_sha256"], "callable configuration digest")
    if not isinstance(row["dependency_versions"], Mapping) or not row[
            "dependency_versions"]:
        raise ObservationError("instrument dependency versions are missing")
    complete = all(row["dependency_versions"].values()) and all(
                   item["implementation_status"] == "pinned"
                   and item["configuration_status"] == "pinned"
                   for item in callable_rows)
    if row["configuration_complete"] is not complete:
        raise ObservationError("instrument completeness disagrees with callable identities")
    supplied = _sha(row.pop("sha256"), "instrument identity digest")
    if supplied != _digest(row):
        raise ObservationError("loaded instrument identity digest mismatch")
    row["sha256"] = supplied
    return _json_copy(row)


def _node_for_cpu(cpu_dir: Path, *, max_entries: int) -> int:
    nodes = []
    for entry in _bounded_entries(cpu_dir, max_entries, f"NUMA mapping for {cpu_dir}"):
        match = re.fullmatch(r"node(\d+)", entry.name)
        if match:
            nodes.append(int(match.group(1)))
    if len(nodes) != 1:
        raise ObservationError(f"{cpu_dir} must expose exactly one NUMA node, got {nodes}")
    return nodes[0]


def read_topology(sysfs_cpu_root: Path, *, max_cpu_ids: int = 4096,
                  max_read_bytes: int = 1024 * 1024,
                  max_entries: int = 8192) -> dict[str, Any]:
    """Read symmetric physical sibling partitions and arbitrary NUMA mappings."""
    entries = _bounded_entries(Path(sysfs_cpu_root), max_entries, "CPU topology")
    cpu_entries = []
    for entry in entries:
        match = re.fullmatch(r"cpu(\d+)", entry.name)
        if match:
            cpu_entries.append((int(match.group(1)), entry))
    if not cpu_entries or len(cpu_entries) > max_cpu_ids:
        raise ObservationError("topology CPU count is empty or exceeds budget")
    rows: dict[int, tuple[frozenset[int], int]] = {}
    for cpu, entry in sorted(cpu_entries):
        siblings = parse_cpu_list(_bounded_text(
            entry / "topology" / "thread_siblings_list", max_read_bytes,
            f"CPU {cpu} siblings").strip(), max_cpu_ids=max_cpu_ids)
        if cpu not in siblings:
            raise ObservationError(f"CPU {cpu} is absent from its sibling set")
        rows[cpu] = (siblings, _node_for_cpu(entry, max_entries=max_entries))
    for cpu, (siblings, _node) in rows.items():
        if set(siblings) - set(rows):
            raise ObservationError(f"CPU {cpu} sibling set references absent CPUs")
        if any(rows[sibling][0] != siblings for sibling in siblings):
            raise ObservationError(f"CPU {cpu} sibling partition is asymmetric")
    groups = sorted({tuple(sorted(siblings)) for siblings, _node in rows.values()})
    if sum(len(group) for group in groups) != len(rows):
        raise ObservationError("physical sibling groups overlap")
    body = {"schema": "epyc.autokernel.cpu_topology.v1",
            "logical_cpus": sorted(rows),
            "physical_sibling_groups": [list(group) for group in groups],
            "cpu_to_physical_group": {
                str(cpu): sorted(rows[cpu][0]) for cpu in sorted(rows)},
            "cpu_to_numa_node": {str(cpu): rows[cpu][1] for cpu in sorted(rows)}}
    return body | {"topology_digest": _digest(body)}


def physical_footprint(logical_cpus: Sequence[int], topology: Mapping[str, Any], *,
                       max_cpu_ids: int = 4096) -> list[int]:
    if not isinstance(logical_cpus, (list, tuple)) or len(logical_cpus) > max_cpu_ids:
        raise ObservationError("logical CPU footprint exceeds budget")
    mapping = topology.get("cpu_to_physical_group")
    if not isinstance(mapping, Mapping):
        raise ObservationError("topology lacks physical sibling mapping")
    result: set[int] = set()
    for cpu in logical_cpus:
        cpu = _integer(cpu, "held logical CPU")
        group = mapping.get(str(cpu))
        if not isinstance(group, list) or not group:
            raise ObservationError(f"held CPU {cpu} is absent from topology")
        result.update(_integer(item, "physical sibling CPU") for item in group)
        if len(result) > max_cpu_ids:
            raise ObservationError("physical CPU footprint exceeds budget")
    return sorted(result)


def _validate_container(value: Any) -> dict[str, Any]:
    row = dict(_closed(value, CONTAINER_FIELDS, "container identity"))
    row["path"] = _text(row["path"], "container path")
    if not row["path"].startswith("/"):
        raise ObservationError("container path must be absolute")
    for key in CONTAINER_FIELDS - {"path"}:
        row[key] = _integer(row[key], f"container {key}")
    return row


def _validate_worker(value: Any) -> dict[str, Any]:
    row = dict(_closed(value, WORKER_FIELDS, "worker binding"))
    row["worker_id"] = _text(row["worker_id"], "worker_id")
    row["worker_incarnation"] = _integer(
        row["worker_incarnation"], "worker_incarnation", 1)
    row["grant_id"] = _text(row["grant_id"], "grant_id")
    row["grant_generation"] = _integer(row["grant_generation"], "grant_generation", 1)
    row["container_identity"] = _validate_container(row["container_identity"])
    return row


def validate_context(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "observation_id", "backend", "instrument_identity_digest",
              "recipe_identity_digest", "clock_domain", "cadence_s", "gap_limit_s",
              "boot_id", "worker_binding", "requested_effective_state", "held_claim",
              "runtime_witness_keys", "required_gpu_dsos", "budgets"}
    value = _closed(value, fields, "observation context")
    if value["schema"] != CONTEXT_SCHEMA:
        raise ObservationError("observation context schema is unsupported")
    backend = value["backend"]
    if backend not in {"cpu", "gpu"}:
        raise ObservationError("observation backend must be cpu or gpu")
    budgets = dict(_closed(value["budgets"], BUDGET_FIELDS, "observation budgets"))
    for key in BUDGET_FIELDS - {
            "phase_ack_timeout_s", "join_timeout_s", "max_probe_duration_s"}:
        budgets[key] = _integer(budgets[key], key, 1)
    for key in ("phase_ack_timeout_s", "join_timeout_s", "max_probe_duration_s"):
        budgets[key] = _finite(budgets[key], key, positive=True)
    cadence = _finite(value["cadence_s"], "cadence_s", positive=True)
    gap = _finite(value["gap_limit_s"], "gap_limit_s", positive=True)
    if gap < cadence:
        raise ObservationError("gap_limit_s cannot be shorter than cadence_s")
    requested = dict(_closed(value["requested_effective_state"], {
        "logical_cpus", "numa_nodes", "thp_mode"}, "requested effective state"))
    claim = dict(_closed(value["held_claim"], {"logical_cpus", "gpu_devices"},
                         "held claim"))
    for name, owner in (("logical_cpus", requested), ("numa_nodes", requested),
                        ("logical_cpus", claim)):
        if not isinstance(owner[name], list) or not owner[name]:
            raise ObservationError(f"{name} must be a non-empty list")
        owner[name] = sorted({_integer(item, name) for item in owner[name]})
        if len(owner[name]) > budgets["max_cpu_ids"]:
            raise ObservationError(f"{name} exceeds CPU/NUMA budget")
    requested["thp_mode"] = _text(requested["thp_mode"], "requested THP mode")
    if not isinstance(claim["gpu_devices"], list):
        raise ObservationError("gpu_devices must be a list")
    claim["gpu_devices"] = sorted({_text(item, "GPU device", maximum=256)
                                    for item in claim["gpu_devices"]})
    if len(claim["gpu_devices"]) > budgets["max_dso_entries"]:
        raise ObservationError("GPU device count exceeds observation budget")
    if not isinstance(value["runtime_witness_keys"], list):
        raise ObservationError("runtime_witness_keys must be a list")
    runtime_keys = [_text(item, "runtime witness key", maximum=256)
                    for item in value["runtime_witness_keys"]]
    if len(runtime_keys) != len(set(runtime_keys)) or len(runtime_keys) > 256:
        raise ObservationError("runtime witness keys are duplicated or exceed budget")
    if not isinstance(value["required_gpu_dsos"], list):
        raise ObservationError("required_gpu_dsos must be a list")
    if len(value["required_gpu_dsos"]) > budgets["max_dso_entries"]:
        raise ObservationError("required GPU DSO count exceeds budget")
    dsos = [_validate_artifact(item) for item in value["required_gpu_dsos"]]
    if backend == "cpu" and (claim["gpu_devices"] or dsos):
        raise ObservationError("CPU observation cannot declare GPU evidence")
    if backend == "gpu" and (not claim["gpu_devices"] or not dsos):
        raise ObservationError("GPU observation requires devices and prepared DSOs")
    return {"schema": CONTEXT_SCHEMA,
            "observation_id": _text(value["observation_id"], "observation_id"),
            "backend": backend,
            "instrument_identity_digest": _sha(
                value["instrument_identity_digest"], "instrument identity digest"),
            "recipe_identity_digest": _sha(
                value["recipe_identity_digest"], "recipe identity digest"),
            "clock_domain": _text(value["clock_domain"], "clock_domain"),
            "cadence_s": cadence, "gap_limit_s": gap,
            "boot_id": _text(value["boot_id"], "boot_id"),
            "worker_binding": _validate_worker(value["worker_binding"]),
            "requested_effective_state": requested, "held_claim": claim,
            "runtime_witness_keys": runtime_keys,
            "required_gpu_dsos": dsos, "budgets": budgets}


def _status_lists(text: str, maximum: int) -> dict[str, list[int]]:
    result = {}
    for key, field in (("cpus_allowed", "Cpus_allowed_list"),
                       ("mems_allowed", "Mems_allowed_list")):
        matches = re.findall(rf"^{field}:\s*(\S+)\s*$", text, re.MULTILINE)
        if len(matches) != 1:
            raise ObservationError(f"process status lacks exactly one {field}")
        result[key] = sorted(parse_cpu_list(matches[0], field, max_cpu_ids=maximum))
    return result


def _key_values(text: str, required: Sequence[str], label: str, *, sum_repeats: bool = False
                ) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.replace(":", " ").split()
        if len(fields) >= 2 and fields[0] in required:
            try:
                value = int(fields[1])
            except ValueError as exc:
                raise ObservationError(f"{label} field {fields[0]} is nonnumeric") from exc
            result[fields[0]] = result.get(fields[0], 0) + value if sum_repeats else value
    if set(required) - set(result) or any(value < 0 for value in result.values()):
        raise ObservationError(f"{label} lacks required non-negative counters")
    return result


def _psi(text: str) -> dict[str, Any]:
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] in {"some", "full"}:
            row = {}
            for item in fields[1:]:
                key, raw = item.split("=", 1)
                row[key] = int(raw) if key == "total" else float(raw)
            if set(row) != {"avg10", "avg60", "avg300", "total"}:
                raise ObservationError("memory PSI row has unsupported fields")
            result[fields[0]] = row
    if set(result) != {"some", "full"}:
        raise ObservationError("memory PSI lacks some/full rows")
    return result


def _numa_maps(text: str, maximum: int) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if len(rows) >= maximum:
            raise ObservationError(f"numa_maps row budget exhausted at {maximum}")
        fields = line.split()
        if len(fields) < 2:
            raise ObservationError("numa_maps row is malformed")
        nodes, page_kb = {}, None
        for item in fields[2:]:
            if match := re.fullmatch(r"N(\d+)=(\d+)", item):
                nodes[match.group(1)] = int(match.group(2))
            elif match := re.fullmatch(r"kernelpagesize_kB=(\d+)", item):
                page_kb = int(match.group(1))
        rows.append({"address": fields[0], "policy": fields[1],
                     "kernel_page_size_kb": page_kb, "pages_by_node": nodes})
    if not rows:
        raise ObservationError("numa_maps is empty")
    return rows


def _mapping_rows(text: str, maximum: int) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if len(rows) >= maximum:
            raise ObservationError(f"maps row budget exhausted at {maximum}")
        fields = line.split(maxsplit=5)
        if len(fields) < 5:
            raise ObservationError("maps row is malformed")
        try:
            major, minor = (int(item, 16) for item in fields[3].split(":", 1))
            inode = int(fields[4])
        except (ValueError, IndexError) as exc:
            raise ObservationError("maps device/inode is malformed") from exc
        rows.append({"device": os.makedev(major, minor), "inode": inode,
                     "path": fields[5] if len(fields) == 6 else None})
    return rows


class FilesystemProbe:
    """Injectable bounded proc/sys reader; no syscall cancellation is claimed."""

    def __init__(self, *, proc_root: Path = Path("/proc"),
                 sysfs_cpu_root: Path = Path("/sys/devices/system/cpu"),
                 boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
                 memory_psi_path: Path = Path("/proc/pressure/memory"),
                 thp_enabled_path: Path = Path(
                     "/sys/kernel/mm/transparent_hugepage/enabled"),
                 cgroup_root: Path = Path("/sys/fs/cgroup"),
                 target_gpu_adapter: Callable[[Mapping[str, Any], Sequence[str],
                                               Mapping[str, Any]], Mapping[str, Any]] | None = None,
                 global_vram_paths: Mapping[str, Path] | None = None) -> None:
        self.proc_root = Path(proc_root)
        self.sysfs_cpu_root = Path(sysfs_cpu_root)
        self.boot_id_path = Path(boot_id_path)
        self.memory_psi_path = Path(memory_psi_path)
        self.thp_enabled_path = Path(thp_enabled_path)
        self.cgroup_root = Path(cgroup_root)
        self.target_gpu_adapter = target_gpu_adapter
        self.global_vram_paths = {
            key: Path(path) for key, path in (global_vram_paths or {}).items()}

    def boot_id(self, budget: Mapping[str, Any]) -> str:
        return _text(_bounded_text(self.boot_id_path, budget["max_read_bytes"],
                                   "boot ID").strip(), "boot ID")

    def topology(self, budget: Mapping[str, Any]) -> dict[str, Any]:
        return read_topology(
            self.sysfs_cpu_root, max_cpu_ids=budget["max_cpu_ids"],
            max_read_bytes=budget["max_read_bytes"],
            max_entries=budget["max_cpu_ids"] * 4)

    def process_identity(self, pid: int, budget: Mapping[str, Any]) -> dict[str, Any]:
        return parse_proc_stat(_bounded_text(
            self.proc_root / str(pid) / "stat", budget["max_read_bytes"],
            f"PID {pid} stat"), pid)

    def _cgroup_path(self, pid: int, budget: Mapping[str, Any]) -> str:
        text = _bounded_text(self.proc_root / str(pid) / "cgroup",
                             budget["max_read_bytes"], f"PID {pid} cgroup")
        rows = [line.split(":", 2) for line in text.splitlines() if line.count(":") == 2]
        unified = [row[2] for row in rows if row[0] == "0" and row[1] == ""]
        if len(unified) != 1:
            raise ObservationError("target lacks exactly one unified cgroup path")
        return unified[0]

    def verify_container(self, pid: int, expected: Mapping[str, Any],
                         budget: Mapping[str, Any]) -> dict[str, Any]:
        expected = _validate_container(expected)
        path = self._cgroup_path(pid, budget)
        if path != expected["path"]:
            raise ObservationError("target cgroup path differs from owned container")
        actual = _stat_identity(self.cgroup_root / path.lstrip("/"))
        actual["path"] = path
        if actual != expected:
            raise ObservationError("target cgroup inode identity differs from owned container")
        return actual

    def _process(self, pid: int, topology: Mapping[str, Any],
                 budget: Mapping[str, Any]) -> dict[str, Any]:
        before = self.process_identity(pid, budget)
        status = _status_lists(_bounded_text(
            self.proc_root / str(pid) / "status", budget["max_read_bytes"],
            f"PID {pid} status"), budget["max_cpu_ids"])
        after = self.process_identity(pid, budget)
        if (before["pid"], before["start_ticks"]) != (after["pid"], after["start_ticks"]):
            raise ObservationError(f"PID {pid} changed identity during process read")
        return after | status | {"physical_affinity_footprint": physical_footprint(
            status["cpus_allowed"], topology, max_cpu_ids=budget["max_cpu_ids"])}

    def _target(self, binding: Mapping[str, Any], context: Mapping[str, Any],
                topology: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, Any]:
        pid = binding["pid"]
        before = self.process_identity(pid, budget)
        if before["start_ticks"] != binding["start_ticks"]:
            raise ObservationError("target PID-start differs before multi-file read")
        container = self.verify_container(
            pid, context["worker_binding"]["container_identity"], budget)
        process = self._process(pid, topology, budget)
        root = self.proc_root / str(pid)
        numa = _numa_maps(_bounded_text(
            root / "numa_maps", budget["max_read_bytes"], "target numa_maps"),
            budget["max_numa_rows"])
        rollup = _key_values(_bounded_text(
            root / "smaps_rollup", budget["max_read_bytes"], "target smaps_rollup"),
            ("Rss", "AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"),
            "target smaps_rollup", sum_repeats=True)
        mappings = _mapping_rows(_bounded_text(
            root / "maps", budget["max_read_bytes"], "target maps"),
            budget["max_map_entries"])
        dso_rows = []
        for expected in context["required_gpu_dsos"]:
            matches = [row for row in mappings
                       if row["device"] == expected["dev"] and row["inode"] == expected["ino"]]
            current, current_error = False, None
            try:
                stat = Path(expected["path"]).stat()
                current = all((stat.st_dev == expected["dev"],
                               stat.st_ino == expected["ino"],
                               stat.st_size == expected["size"],
                               stat.st_mtime_ns == expected["mtime_ns"],
                               stat.st_ctime_ns == expected["ctime_ns"]))
                if not current:
                    current_error = "prepared artifact metadata changed"
            except OSError as exc:
                current_error = f"{type(exc).__name__}: {exc}"
            dso_rows.append({"artifact_identity_sha256": expected["identity_sha256"],
                             "prepared_path": expected["path"],
                             "mapped_device": expected["dev"],
                             "mapped_inode": expected["ino"],
                             "matched_mapping_count": len(matches),
                             "matched_paths": sorted({row["path"] for row in matches
                                                      if row["path"] is not None}),
                             "prepared_identity_current": current,
                             "current_identity_error": current_error})
        fds = _bounded_entries(root / "fd", budget["max_fd_entries"], "target FDs")
        kfd_fds = []
        for entry in fds:
            try:
                target = os.readlink(entry)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ObservationError(f"cannot read target FD {entry}: {exc}") from exc
            if target == "/dev/kfd":
                kfd_fds.append(entry.name)
        after = self.process_identity(pid, budget)
        if (after["pid"], after["start_ticks"]) != (pid, binding["start_ticks"]):
            raise ObservationError("target PID-start changed during multi-file read")
        return process | {"container_identity": container, "numa_maps": numa,
                          "smaps_rollup_kb": rollup, "required_gpu_dsos": dso_rows,
                          "kfd_fds": sorted(kfd_fds)}

    def _gpu(self, binding: Mapping[str, Any] | None, context: Mapping[str, Any],
             budget: Mapping[str, Any]) -> dict[str, Any]:
        global_vram: dict[str, int | None] = {}
        for device in context["held_claim"]["gpu_devices"]:
            path = self.global_vram_paths.get(device)
            if path is None:
                global_vram[device] = None
                continue
            try:
                global_vram[device] = _integer(int(_bounded_text(
                    path, budget["max_read_bytes"], f"global VRAM {device}").strip()),
                    f"global VRAM {device}")
            except (ObservationError, ValueError):
                global_vram[device] = None
        unknown = {"status": "unknown", "pid": None, "start_ticks": None,
                   "allocated_bytes": {}, "evidence_ref": None,
                   "reason": "target_not_attached"}
        if binding is None:
            return {"global_vram_bytes": global_vram, "target_attribution": unknown}
        pid = binding["pid"]
        if self.target_gpu_adapter is None:
            return {"global_vram_bytes": global_vram,
                    "target_attribution": unknown | {"pid": pid,
                        "start_ticks": binding["start_ticks"],
                        "reason": "target_gpu_adapter_unavailable"}}
        try:
            before = self.process_identity(pid, budget)
            supplied = _closed(self.target_gpu_adapter(
                binding, tuple(context["held_claim"]["gpu_devices"]), budget),
                {"status", "allocated_bytes", "evidence_ref", "reason"},
                "target GPU adapter result")
            after = self.process_identity(pid, budget)
            if before["start_ticks"] != binding["start_ticks"] or after != before:
                raise ObservationError("target PID-start changed around GPU adapter read")
            if supplied["status"] != "observed":
                raise ObservationError(_text(supplied["reason"], "GPU adapter reason"))
            if not isinstance(supplied["allocated_bytes"], Mapping) or set(
                    supplied["allocated_bytes"]) != set(context["held_claim"]["gpu_devices"]):
                raise ObservationError("target GPU allocation device set differs")
            allocated = {device: _integer(value, f"target allocation {device}")
                         for device, value in supplied["allocated_bytes"].items()}
            target = {"status": "observed", "pid": pid,
                      "start_ticks": binding["start_ticks"], "allocated_bytes": allocated,
                      "evidence_ref": _text(supplied["evidence_ref"],
                                            "target GPU evidence ref"), "reason": None}
        except Exception as exc:
            target = {"status": "unknown", "pid": pid,
                      "start_ticks": binding["start_ticks"], "allocated_bytes": {},
                      "evidence_ref": None,
                      "reason": f"{type(exc).__name__}: {exc}"}
        return {"global_vram_bytes": global_vram, "target_attribution": target}

    def capture(self, *, context: Mapping[str, Any], topology: Mapping[str, Any],
                target_binding: Mapping[str, Any] | None) -> dict[str, Any]:
        budget = context["budgets"]
        subprobe_errors = []
        processes, census_gaps = [], []
        try:
            entries = _bounded_entries(self.proc_root, budget["max_proc_entries"],
                                       "procfs root")
            process_dirs = sorted((entry for entry in entries if entry.name.isdigit()),
                                  key=lambda path: int(path.name))
            if len(process_dirs) > budget["max_processes"]:
                raise ObservationError(
                    f"process census budget exhausted at {budget['max_processes']}")
            for path in process_dirs:
                pid = int(path.name)
                try:
                    processes.append(self._process(pid, topology, budget))
                except Exception as exc:
                    census_gaps.append({"pid": pid,
                                        "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            subprobe_errors.append({"probe": "process_census",
                                    "error": f"{type(exc).__name__}: {exc}"})
        target, target_error = None, None
        if target_binding is not None:
            try:
                target = self._target(target_binding, context, topology, budget)
            except Exception as exc:
                target_error = f"{type(exc).__name__}: {exc}"
        def attempt(name, function):
            try:
                return function()
            except Exception as exc:
                subprobe_errors.append({"probe": name,
                                        "error": f"{type(exc).__name__}: {exc}"})
                return None
        memory = attempt("memory", lambda: _key_values(_bounded_text(
            self.proc_root / "meminfo", budget["max_read_bytes"], "meminfo"),
            ("MemAvailable", "SwapFree", "SwapTotal"), "meminfo"))
        vmstat = attempt("vmstat", lambda: _key_values(_bounded_text(
            self.proc_root / "vmstat", budget["max_read_bytes"], "vmstat"),
            ("pswpin", "pswpout"), "vmstat"))
        psi = attempt("memory_psi", lambda: _psi(_bounded_text(
            self.memory_psi_path, budget["max_read_bytes"], "memory PSI")))
        thp_raw = attempt("thp", lambda: _bounded_text(
            self.thp_enabled_path, budget["max_read_bytes"],
            "THP effective state").strip())
        selected = re.findall(r"\[([^]]+)\]", thp_raw) if thp_raw is not None else []
        if thp_raw is not None and len(selected) != 1:
            subprobe_errors.append({"probe": "thp",
                                    "error": "effective state lacks one selected mode"})
        effective = None
        if target is not None and len(selected) == 1:
            requested = context["requested_effective_state"]
            effective = {
                "logical_cpus": target["cpus_allowed"],
                "numa_nodes": target["mems_allowed"], "thp_mode": selected[0],
                "logical_cpus_match": target["cpus_allowed"] == requested["logical_cpus"],
                "numa_nodes_match": target["mems_allowed"] == requested["numa_nodes"],
                "thp_mode_match": selected[0] == requested["thp_mode"],
                "recipe_identity_digest": context["recipe_identity_digest"],
                "instrument_identity_digest": context["instrument_identity_digest"]}
        gpu = (attempt("gpu", lambda: self._gpu(target_binding, context, budget))
               if context["backend"] == "gpu" else {
                   "global_vram_bytes": {},
                   "target_attribution": {"status": "not_applicable", "pid": None,
                                          "start_ticks": None, "allocated_bytes": {},
                                          "evidence_ref": None,
                                          "reason": "CPU arm"}})
        return {"processes": processes, "census_gaps": census_gaps,
                "subprobe_errors": subprobe_errors,
                "target": target, "target_error": target_error,
                "memory": memory, "vmstat": vmstat, "memory_psi": psi,
                "thp_effective": ({"raw": thp_raw, "selected": selected[0]}
                                  if len(selected) == 1 else None),
                "effective_readback": effective, "gpu": gpu}


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


class ObservationSession:
    """Single-use async sampler with bounded marker waits and frozen shutdown."""

    def __init__(self, context: Mapping[str, Any], *, probe: FilesystemProbe,
                 owned_identity_resolver: Callable[[int], Mapping[str, Any]],
                 foreign_verifier: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
                 runtime_verifier: Callable[[str, Mapping[str, Any] | None],
                                            Mapping[str, Any]] | None = None,
                 monotonic: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], str] = _now_utc,
                 record_callback: Callable[[Mapping[str, Any]], Any] | None = None,
                 phase_notice: Callable[[str, Mapping[str, Any],
                                         Mapping[str, Any] | None], None] | None = None) -> None:
        if not callable(owned_identity_resolver):
            raise ObservationError("a trusted owned identity resolver is required")
        self.context = validate_context(context)
        self.probe = probe
        self.owned_identity_resolver = owned_identity_resolver
        self.foreign_verifier = foreign_verifier
        self.runtime_verifier = runtime_verifier
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self.record_callback = record_callback
        if phase_notice is not None and not callable(phase_notice):
            raise ObservationError("phase notification must be callable")
        self.phase_notice = phase_notice
        self.artifact_receipt: Any = None
        self.callback_error: str | None = None
        self._condition = threading.Condition()
        self._pending: deque[dict[str, Any]] = deque()
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopping = False
        self._accepting = False
        self._finished = False
        self._shutdown_resolved = False
        self._phase: str | None = None
        self._scheduled = 0
        self._retained_bytes = 0
        self._dropped_samples = 0
        self._dropped_markers = 0
        self._completed_reads = 0
        self._failed_reads = 0
        self._oversized_samples = 0
        self._reader_seconds = 0.0
        self._active_reads = 0
        self._reader_cost_status = "complete"
        self._unresolved_read_count = 0
        self._samples: list[dict[str, Any]] = []
        self._boundaries: list[dict[str, Any]] = []
        self._issues: list[dict[str, Any]] = []
        self._topology: dict[str, Any] | None = None
        self._target_pid: int | None = None
        self._target_binding: dict[str, Any] | None = None
        self._start_mono: float | None = None
        self._start_wall: str | None = None
        self._end_mono: float | None = None
        self._end_wall: str | None = None
        self._final: dict[str, Any] | None = None

    @property
    def shutdown_resolved(self) -> bool:
        with self._condition:
            return self._shutdown_resolved

    @property
    def successor_permitted(self) -> bool:
        with self._condition:
            return self._finished and self._shutdown_resolved

    def reconcile_shutdown(self) -> bool:
        """Recheck ownership after a previously stuck reader returns; never alter evidence."""
        with self._condition:
            if not self._finished or self._thread is None:
                raise ObservationError("shutdown reconciliation requires a finished session")
            if not self._thread.is_alive():
                self._shutdown_resolved = True
            return self._shutdown_resolved

    def _time_pair(self) -> tuple[float, str]:
        return (_finite(self.monotonic(), "monotonic clock"),
                _utc(self.wall_clock(), "wall clock"))

    def _issue_locked(self, code: str, detail: str, **counts: Any) -> None:
        self._issues.append({"code": code, "detail": detail, "counts": counts})

    def _enqueue_locked(self, phase: str, kind: str,
                        marker_mono: float, marker_wall: str,
                        marker_label: str | None = None) -> threading.Event | None:
        budgets = self.context["budgets"]
        if self._scheduled >= budgets["max_samples"]:
            self._record_marker_drop_locked(
                "sample_budget_exhausted", "sample budget exhausted",
                scheduled=self._scheduled, maximum=budgets["max_samples"])
            return None
        if len(self._pending) >= budgets["max_pending_markers"]:
            self._record_marker_drop_locked(
                "marker_budget_exhausted", "pending marker budget exhausted",
                pending=len(self._pending), maximum=budgets["max_pending_markers"])
            return None
        done = threading.Event()
        self._pending.append({"phase": phase, "kind": kind,
                              "marker_monotonic_s": marker_mono,
                              "marker_at": marker_wall, "marker_label": marker_label,
                              "done": done})
        self._scheduled += 1
        self._condition.notify()
        return done

    def _record_marker_drop_locked(self, code: str, detail: str, **counts: Any) -> None:
        ceiling = 2**63 - 1
        self._dropped_markers = min(ceiling, self._dropped_markers + 1)
        counts |= {"dropped_marker_count": self._dropped_markers,
                   "counter_saturated": self._dropped_markers == ceiling}
        existing = next((row for row in self._issues if row["code"] == code), None)
        if existing is None:
            self._issue_locked(code, detail, **counts)
        else:
            existing["counts"] = counts

    def _wait_marker(self, done: threading.Event | None, phase: str) -> None:
        if done is None:
            return
        timeout = self.context["budgets"]["phase_ack_timeout_s"]
        if not done.wait(timeout):
            with self._condition:
                if self._accepting:
                    self._issue_locked("phase_sample_timeout",
                                       f"{phase} marker was not sampled within bound",
                                       timeout_s=timeout)

    def start(self, phase: str = "setup") -> None:
        if phase != "setup":
            raise ObservationError("whole-lifecycle observation must begin at setup")
        start_mono, start_wall = self._time_pair()
        with self._condition:
            if self._started:
                raise ObservationError("observation session is single-use")
            self._started = self._accepting = True
            self._phase = phase
            self._start_mono, self._start_wall = start_mono, start_wall
            self._boundaries.append({"phase": phase, "monotonic_s": self._start_mono,
                                     "captured_at": self._start_wall})
            done = self._enqueue_locked(phase, "boundary", self._start_mono,
                                        self._start_wall)
            self._thread = threading.Thread(
                target=self._run, name="autokernel-lifecycle-observer", daemon=True)
            self._thread.start()
        self._wait_marker(done, phase)

    def attach_target(self, pid: int) -> None:
        pid = _integer(pid, "target PID", 1)
        mono, wall = self._time_pair()
        with self._condition:
            if not self._started or self._stopping or self._target_pid is not None:
                raise ObservationError("target attachment requires one active unbound session")
            self._target_pid = pid
            phase = self._phase
            assert phase is not None
            done = self._enqueue_locked(phase, "target_attached", mono, wall)
        self._wait_marker(done, phase)

    def phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise ObservationError(f"unsupported observation phase {phase!r}")
        mono, wall = self._time_pair()
        with self._condition:
            if not self._started or self._stopping or self._phase is None:
                raise ObservationError("phase transition requires an active observation")
            if PHASES.index(phase) <= PHASES.index(self._phase):
                raise ObservationError("observation phases must advance")
            skipped = PHASES[PHASES.index(self._phase) + 1:PHASES.index(phase)]
            if skipped:
                self._issue_locked("phase_marker_gap",
                                   "exception path skipped lifecycle markers",
                                   missing=list(skipped))
            self._phase = phase
            self._boundaries.append({"phase": phase, "monotonic_s": mono,
                                     "captured_at": wall})
            done = self._enqueue_locked(phase, "boundary", mono, wall)
        self._wait_marker(done, phase)
        if self.phase_notice is not None:
            with self._condition:
                target = _json_copy(self._target_binding)
            # Notification only, never a witness verdict. No observer lock is held
            # while the inherited parent authority captures its own live readback.
            self.phase_notice(phase, {"phase": phase, "monotonic_s": mono,
                                     "captured_at": wall}, target)

    def checkpoint(self, label: str) -> None:
        """Request one bounded sample inside the current lifecycle phase."""
        label = _text(label, "checkpoint label", maximum=256)
        mono, wall = self._time_pair()
        with self._condition:
            if not self._started or self._stopping or self._phase is None:
                raise ObservationError("checkpoint requires an active observation")
            phase = self._phase
            done = self._enqueue_locked(phase, "checkpoint", mono, wall, label)
        self._wait_marker(done, phase)
        if label == "measurement_end" and phase == "measurement" and self.phase_notice is not None:
            with self._condition:
                target = _json_copy(self._target_binding)
            self.phase_notice("measurement_end", {
                "phase": phase, "marker_label": label,
                "monotonic_s": mono, "captured_at": wall}, target)

    def _resolve_target(self, pid: int) -> dict[str, Any]:
        supplied = self.owned_identity_resolver(pid)
        fields = {"pid", "start_ticks", "boot_id", "worker_binding", "binding_ref"}
        supplied = _closed(supplied, fields, "trusted owned process binding")
        row = {"pid": _integer(supplied["pid"], "owned PID", 1),
               "start_ticks": _integer(supplied["start_ticks"], "owned start ticks", 1),
               "boot_id": _text(supplied["boot_id"], "owned boot ID"),
               "worker_binding": _validate_worker(supplied["worker_binding"]),
               "binding_ref": _text(supplied["binding_ref"], "owned binding reference")}
        if row["pid"] != pid or row["boot_id"] != self.context["boot_id"]:
            raise ObservationError("trusted process binding differs from PID/boot")
        if row["worker_binding"] != self.context["worker_binding"]:
            raise ObservationError("trusted process binding differs from worker/container/grant")
        return row

    def note_hook_failure(self, method: str, exc: Exception) -> None:
        """Retain a launcher-hook diagnostic without raising into owned cleanup."""
        with self._condition:
            if self._accepting:
                self._issue_locked("serving_hook_failed", f"{method}: {type(exc).__name__}: {exc}")

    def _verify_foreign(self, process: Mapping[str, Any]) -> dict[str, Any]:
        unknown = {"status": "unknown", "kind": None, "evidence_ref": None,
                   "reason": "verifier_unavailable"}
        if self.foreign_verifier is None:
            return unknown
        try:
            row = dict(_closed(self.foreign_verifier(process),
                               {"status", "kind", "evidence_ref", "reason"},
                               "foreign verification"))
            if row["status"] not in {"verified", "unknown"}:
                raise ObservationError("foreign verification status is unsupported")
            if row["status"] == "verified":
                if row["kind"] not in FOREIGN_KINDS:
                    raise ObservationError("verified foreign kind is unsupported")
                row["evidence_ref"] = _text(row["evidence_ref"], "foreign evidence ref")
                row["reason"] = None
            else:
                row = unknown | {"reason": _text(row["reason"], "unknown reason")}
            return row
        except Exception as exc:
            return unknown | {"reason": f"verifier_error:{type(exc).__name__}"}

    def _verify_runtime(self, key: str, target: Mapping[str, Any] | None) -> dict[str, Any]:
        if self.runtime_verifier is None:
            return {"key": key, "status": "unknown", "evidence_ref": None,
                    "reason": "verifier_unavailable"}
        try:
            row = dict(_closed(self.runtime_verifier(key, target),
                               {"status", "evidence_ref", "reason"},
                               "runtime verification"))
            if row["status"] not in set(RUNTIME_WITNESS_STATES) - {"unknown"}:
                raise ObservationError("runtime verification status is unsupported")
            if row["status"] == "fired_under_target":
                row["evidence_ref"] = _text(row["evidence_ref"], "runtime evidence ref")
            return {"key": key, **row}
        except Exception as exc:
            return {"key": key, "status": "unknown", "evidence_ref": None,
                    "reason": f"verifier_error:{type(exc).__name__}"}

    def _capture_marker(self, marker: Mapping[str, Any]) -> dict[str, Any]:
        read_start_mono, read_start_wall = self._time_pair()
        error = None
        try:
            topology = self._topology
            if topology is None:
                if self.probe.boot_id(self.context["budgets"]) != self.context["boot_id"]:
                    raise ObservationError("probe boot ID differs from observation binding")
                topology = self.probe.topology(self.context["budgets"])
            target = self._target_binding
            if target is None and self._target_pid is not None:
                target = self._resolve_target(self._target_pid)
            payload = self.probe.capture(context=self.context, topology=topology,
                                         target_binding=target)
            owned = None if target is None else (target["pid"], target["start_ticks"])
            for process in payload["processes"]:
                identity = (process["pid"], process["start_ticks"])
                process["ownership"] = "owned" if identity == owned else "foreign"
                process["foreign_verification"] = (
                    None if identity == owned else self._verify_foreign(process))
            payload["runtime_witnesses"] = [
                self._verify_runtime(key, payload["target"])
                for key in self.context["runtime_witness_keys"]]
            gpu_unknown = (self.context["backend"] == "gpu" and target is not None
                           and (payload["gpu"] is None or payload["gpu"][
                               "target_attribution"]["status"] != "observed"))
            if (payload["census_gaps"] or payload["subprobe_errors"]
                    or payload["target_error"] is not None or gpu_unknown):
                status = "unknown"
                error = "one or more bounded subprobes are unknown; partial facts retained"
            else:
                status = "observed"
        except Exception as exc:
            topology = self._topology
            target = self._target_binding
            payload = {"processes": [], "census_gaps": [], "subprobe_errors": [],
                       "target": None,
                       "target_error": None, "memory": None, "vmstat": None,
                       "memory_psi": None, "thp_effective": None,
                       "effective_readback": None, "gpu": None,
                       "runtime_witnesses": []}
            status = "unknown"
            error = f"{type(exc).__name__}: {exc}"
        read_end_mono, read_end_wall = self._time_pair()
        if read_end_mono < read_start_mono:
            status, error = "unknown", "ObservationError: probe clock moved backwards"
        sample = {"schema": SAMPLE_SCHEMA, "status": status,
                  "phase": marker["phase"], "kind": marker["kind"],
                  "marker_label": marker["marker_label"],
                  "marker_monotonic_s": marker["marker_monotonic_s"],
                  "marker_at": marker["marker_at"],
                  "read_started_monotonic_s": read_start_mono,
                  "read_ended_monotonic_s": read_end_mono,
                  "read_started_at": read_start_wall, "read_ended_at": read_end_wall,
                  "queue_delay_s": read_start_mono - marker["marker_monotonic_s"],
                  "read_duration_s": read_end_mono - read_start_mono,
                  **payload, "error": error}
        sample["probe_over_budget"] = (
            sample["read_duration_s"] > self.context["budgets"]["max_probe_duration_s"])
        sample["topology"] = topology if self._topology is None else None
        sample["resolved_target_binding"] = target if self._target_binding is None else None
        return sample

    def _commit(self, sample: dict[str, Any], marker: Mapping[str, Any]) -> None:
        sample_bytes = len(_canonical(sample))
        with self._condition:
            self._active_reads -= 1
            if self._accepting:
                self._completed_reads += 1
                self._reader_seconds += max(0.0, sample["read_duration_s"])
                if sample["status"] == "unknown":
                    self._failed_reads += 1
                if sample["topology"] is not None:
                    self._topology = sample["topology"]
                    footprint = physical_footprint(
                        self.context["held_claim"]["logical_cpus"], self._topology,
                        max_cpu_ids=self.context["budgets"]["max_cpu_ids"])
                    self.context["held_claim"]["physical_cpus"] = footprint
                if sample["resolved_target_binding"] is not None:
                    self._target_binding = sample["resolved_target_binding"]
                sample.pop("topology")
                sample.pop("resolved_target_binding")
                if sample["status"] == "unknown":
                    self._issue_locked("sample_failed", sample["error"] or "unknown sample")
                if sample["probe_over_budget"]:
                    self._issue_locked("probe_duration_exhausted",
                                       "probe exceeded configured duration observation",
                                       duration_s=sample["read_duration_s"],
                                       maximum_s=self.context["budgets"][
                                           "max_probe_duration_s"])
                maximum = self.context["budgets"]["max_retained_bytes"]
                if self._retained_bytes + sample_bytes > maximum:
                    self._dropped_samples += 1
                    self._oversized_samples += 1
                    existing = next((row for row in self._issues
                                     if row["code"] == "retained_byte_budget_exhausted"), None)
                    counts = {"retained_bytes": self._retained_bytes,
                              "maximum_bytes": maximum,
                              "dropped_samples": self._dropped_samples,
                              "last_sample_bytes": sample_bytes}
                    if existing is None:
                        self._issue_locked("retained_byte_budget_exhausted",
                                           "sample was not retained", **counts)
                    else:
                        existing["counts"] = counts
                else:
                    self._samples.append(sample)
                    self._retained_bytes += sample_bytes
            marker["done"].set()

    def _run(self) -> None:
        cadence = self.context["cadence_s"]
        due = None
        while True:
            with self._condition:
                if self._pending:
                    marker = self._pending.popleft()
                    due = None
                elif self._stopping:
                    return
                else:
                    marker = None
                periodic_phase = self._phase
            if marker is None:
                mono, wall = self._time_pair()
                with self._condition:
                    if self._stopping:
                        return
                    # A boundary/checkpoint may arrive during the unlocked clock
                    # read. Do not queue an old-phase sample behind that marker.
                    if self._pending or self._phase != periodic_phase:
                        due = None
                        continue
                    if periodic_phase is None:
                        raise ObservationError("active observer has no phase")
                    if due is None:
                        due = mono + cadence
                    remaining = due - mono
                    if remaining > 0:
                        # Keep the same due time through empty early/spurious
                        # wakeups: they neither sample early nor postpone forever.
                        self._condition.wait(remaining)
                        continue
                    due = None
                    done = self._enqueue_locked(periodic_phase, "periodic", mono, wall)
                    marker = self._pending.popleft() if done is not None else None
                if marker is None:
                    continue
            with self._condition:
                self._active_reads += 1
            sample = self._capture_marker(marker)
            self._commit(sample, marker)

    def finish(self) -> dict[str, Any]:
        with self._condition:
            if not self._started or self._finished:
                raise ObservationError("finish requires one active observation")
            if self._phase != "teardown":
                self._issue_locked("teardown_phase_missing",
                                   "finish occurred before teardown marker")
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread
        assert thread is not None
        thread.join(self.context["budgets"]["join_timeout_s"])
        end_mono, end_wall = self._time_pair()
        with self._condition:
            self._shutdown_resolved = not thread.is_alive()
            self._accepting = False
            self._reader_cost_status = (
                "complete" if self._shutdown_resolved else "unknown")
            self._unresolved_read_count = self._active_reads
            if not self._shutdown_resolved:
                self._issue_locked(
                    "observer_shutdown_unresolved",
                    "reader remains unresolved; containing owner must refuse a successor")
            self._end_mono, self._end_wall = end_mono, end_wall
            self._finished = True
        # No writer can commit after accepting=False. Build/canonicalize the bounded
        # snapshot without holding the short state lock used by ownership queries.
        final = validate_observation(self._build_record())
        with self._condition:
            self._final = final
            final = _json_copy(final)
        if self.record_callback is not None:
            try:
                self.artifact_receipt = self.record_callback(_json_copy(final))
            except Exception as exc:
                self.callback_error = f"{type(exc).__name__}: {exc}"
        return final

    def record(self) -> dict[str, Any]:
        with self._condition:
            if self._final is None:
                raise ObservationError("record is unavailable before finish")
            return _json_copy(self._final)

    def _build_record(self) -> dict[str, Any]:
        samples = _json_copy(self._samples)
        intervals, interval_issues = _intervals(
            samples, set(self.context["held_claim"].get("physical_cpus", [])),
            self.context["gap_limit_s"])
        issues = _json_copy(self._issues) + interval_issues
        observed_phases = {sample["phase"] for sample in samples
                           if sample["status"] == "observed"}
        missing = [phase for phase in PHASES if phase not in observed_phases]
        if missing:
            issues.append({"code": "missing_phase_samples",
                           "detail": "one or more lifecycle phases have no usable sample",
                           "counts": {"missing": missing}})
        duration = max(0.0, (self._end_mono or 0.0) - (self._start_mono or 0.0))
        reader_s = self._reader_seconds
        load_start = next((row for row in self._boundaries if row["phase"] == "load"), None)
        health_end = next((row for row in self._boundaries if row["phase"] == "health"), None)
        held_claim = _json_copy(self.context["held_claim"])
        held_claim.setdefault("physical_cpus", None)
        body = {"schema": OBSERVATION_SCHEMA, "detector_version": DETECTOR_VERSION,
                "observation_id": self.context["observation_id"],
                "backend": self.context["backend"],
                "instrument_identity_digest": self.context["instrument_identity_digest"],
                "recipe_identity_digest": self.context["recipe_identity_digest"],
                "clock_domain": self.context["clock_domain"],
                "cadence_s": self.context["cadence_s"],
                "gap_limit_s": self.context["gap_limit_s"],
                "started_monotonic_s": self._start_mono, "ended_monotonic_s": self._end_mono,
                "started_at": self._start_wall, "ended_at": self._end_wall,
                "boot_id": self.context["boot_id"],
                "worker_binding": self.context["worker_binding"],
                "held_claim": held_claim,
                "requested_effective_state": self.context["requested_effective_state"],
                "budgets": self.context["budgets"], "topology": self._topology,
                "target_binding": self._target_binding,
                "phase_boundaries": _json_copy(self._boundaries),
                "load_window": {"start": load_start, "end": health_end,
                                "status": "observed" if load_start and health_end else "unknown"},
                "samples": samples, "intervals": intervals,
                "observer_cost": {"reader_seconds": reader_s,
                                  "reader_cost_status": self._reader_cost_status,
                                  "completed_read_count": self._completed_reads,
                                  "failed_read_count": self._failed_reads,
                                  "oversized_sample_count": self._oversized_samples,
                                  "unresolved_read_count": self._unresolved_read_count,
                                  "observation_seconds": duration,
                                  "fraction": reader_s / duration if duration else None,
                                  "sample_count": len(samples),
                                  "sample_budget": self.context["budgets"]["max_samples"],
                                  "dropped_marker_count": self._dropped_markers,
                                  "dropped_sample_count": self._dropped_samples,
                                  "retained_bytes": self._retained_bytes,
                                  "retained_byte_budget": self.context["budgets"][
                                      "max_retained_bytes"]},
                "shutdown": {"status": "resolved" if self._shutdown_resolved else "unresolved",
                             "successor_permitted": self._shutdown_resolved,
                             "late_writes_accepted": False},
                "completeness": "complete" if not issues else "unknown", "issues": issues,
                "residency": _residency(self.context["backend"], samples),
                "verdict": {"status": "not_evaluated",
                            "reason": "observation is separate from owning protocol"}}
        return body | {"content_sha256": _digest(body)}


def _intervals(samples: Sequence[Mapping[str, Any]], claimed: set[int], gap_limit: float
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    intervals, issues = [], []
    for before, after in zip(samples, samples[1:], strict=False):
        elapsed = after["read_started_monotonic_s"] - before["read_ended_monotonic_s"]
        row = {"start_monotonic_s": before["read_ended_monotonic_s"],
               "end_monotonic_s": after["read_started_monotonic_s"],
               "start_at": before["read_ended_at"], "end_at": after["read_started_at"],
               "elapsed_s": elapsed, "phase": after["phase"], "status": "observed",
               "census_changes": {"born": [], "disappeared": []},
               "potential_foreign_overlap": [], "memory_delta": None}
        if elapsed < 0 or elapsed > gap_limit:
            row["status"] = "unknown"
            code = "overlapping_probe_windows" if elapsed < 0 else "sampling_gap"
            issues.append({"code": code, "detail": "interval coverage is not continuous",
                           "counts": {"elapsed_s": elapsed, "gap_limit_s": gap_limit}})
        if before["status"] != "observed" or after["status"] != "observed":
            row["status"] = "unknown"
        else:
            prior = {(item["pid"], item["start_ticks"]): item
                     for item in before["processes"]}
            current = {(item["pid"], item["start_ticks"]): item
                       for item in after["processes"]}
            born, disappeared = sorted(set(current) - set(prior)), sorted(set(prior) - set(current))
            row["census_changes"] = {
                "born": [{"pid": pid, "start_ticks": start} for pid, start in born],
                "disappeared": [{"pid": pid, "start_ticks": start}
                                for pid, start in disappeared]}
            reused = sorted(set(pid for pid, _ in prior) & set(pid for pid, _ in current)
                            & {pid for pid, start in prior
                               if (pid, start) not in current})
            if born or disappeared:
                row["status"] = "unknown"
                issues.append({"code": "process_census_changed",
                               "detail": "born/disappeared identities prevent silent omission",
                               "counts": {"born": len(born), "disappeared": len(disappeared)}})
            if reused:
                issues.append({"code": "pid_reuse", "detail": "PID start identity changed",
                               "counts": {"pids": reused}})
            for identity in sorted(set(prior) & set(current)):
                left, right = prior[identity], current[identity]
                delta = right["cpu_ticks"] - left["cpu_ticks"]
                if delta < 0:
                    row["status"] = "unknown"
                    issues.append({"code": "counter_reset", "detail": "CPU ticks regressed",
                                   "counts": {"pid": identity[0]}})
                    continue
                overlap = sorted(claimed.intersection(right["physical_affinity_footprint"]))
                if delta <= 0 or right["ownership"] == "owned" or not overlap:
                    continue
                row["potential_foreign_overlap"].append({
                    "pid": identity[0], "start_ticks": identity[1],
                    "process_total_cpu_tick_delta": delta,
                    "potential_physical_claim_overlap": overlap,
                    "verification": right["foreign_verification"]})
            row["memory_delta"] = {
                "mem_available_kb": after["memory"]["MemAvailable"]
                - before["memory"]["MemAvailable"],
                "swap_free_kb": after["memory"]["SwapFree"] - before["memory"]["SwapFree"],
                "pswpin": after["vmstat"]["pswpin"] - before["vmstat"]["pswpin"],
                "pswpout": after["vmstat"]["pswpout"] - before["vmstat"]["pswpout"],
                "psi_some_total": after["memory_psi"]["some"]["total"]
                - before["memory_psi"]["some"]["total"],
                "psi_full_total": after["memory_psi"]["full"]["total"]
                - before["memory_psi"]["full"]["total"]}
            if any(row["memory_delta"][key] < 0 for key in (
                    "pswpin", "pswpout", "psi_some_total", "psi_full_total")):
                row["status"] = "unknown"
                issues.append({"code": "counter_reset",
                               "detail": "memory/PSI counter regressed", "counts": {}})
        intervals.append(row)
    return intervals, issues


def _residency(backend: str, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if backend == "cpu":
        return {"status": "not_applicable", "measurement_samples": 0,
                "positive_samples": 0, "reason": "CPU arm"}
    rows = [row for row in samples
            if row["phase"] == "measurement" and row["status"] == "observed"]
    positives = 0
    unknown_attribution = False
    unknown_dso_identity = False
    for row in rows:
        target = row["target"]
        if not target or any(not item["prepared_identity_current"]
                             for item in target["required_gpu_dsos"]):
            unknown_dso_identity = True
        dso = bool(target) and bool(target["required_gpu_dsos"]) and all(
            item["matched_mapping_count"] > 0 and item["prepared_identity_current"]
            for item in target["required_gpu_dsos"])
        attribution = row["gpu"]["target_attribution"]
        if attribution["status"] != "observed":
            unknown_attribution = True
            continue
        allocated = any(value > 0 for value in attribution["allocated_bytes"].values())
        kfd = bool(target and target["kfd_fds"])
        positives += int(dso and allocated and kfd)
    status = ("unknown" if len(rows) < 2 or unknown_attribution or unknown_dso_identity else
              "observed" if positives else "not_observed")
    return {"status": status, "measurement_samples": len(rows),
            "positive_samples": positives,
            "reason": "exact mapped DSO plus target KFD allocation and FD required"}


def _validate_process_row(value: Any, *, target: bool = False) -> None:
    fields = {"pid", "comm", "cpu_ticks", "start_ticks", "processor", "cpus_allowed",
              "mems_allowed", "physical_affinity_footprint"}
    if target:
        fields |= {"container_identity", "numa_maps", "smaps_rollup_kb",
                   "required_gpu_dsos", "kfd_fds"}
    else:
        fields |= {"ownership", "foreign_verification"}
    row = _closed(value, fields, "target process" if target else "process census row")
    for key in ("pid", "cpu_ticks", "start_ticks", "processor"):
        _integer(row[key], f"process {key}", 1 if key in {"pid", "start_ticks"} else 0)
    if target:
        _validate_container(row["container_identity"])
        for numa in row["numa_maps"]:
            _closed(numa, {"address", "policy", "kernel_page_size_kb", "pages_by_node"},
                    "target NUMA row")
        _closed(row["smaps_rollup_kb"],
                {"Rss", "AnonHugePages", "ShmemPmdMapped", "FilePmdMapped"},
                "target smaps rollup")
        for dso in row["required_gpu_dsos"]:
            _closed(dso, {"artifact_identity_sha256", "prepared_path", "mapped_device",
                          "mapped_inode", "matched_mapping_count", "matched_paths",
                          "prepared_identity_current", "current_identity_error"},
                    "loaded DSO evidence")
    else:
        if row["ownership"] not in {"owned", "foreign"}:
            raise ObservationError("process ownership is unsupported")
        if row["foreign_verification"] is not None:
            verification = _closed(row["foreign_verification"],
                                   {"status", "kind", "evidence_ref", "reason"},
                                   "foreign verification")
            if verification["status"] not in {"verified", "unknown"}:
                raise ObservationError("foreign verification status is unsupported")


def _validate_sample(value: Any) -> None:
    fields = {"schema", "status", "phase", "kind", "marker_label",
              "marker_monotonic_s", "marker_at",
              "read_started_monotonic_s", "read_ended_monotonic_s", "read_started_at",
              "read_ended_at", "queue_delay_s", "read_duration_s", "processes",
              "census_gaps", "subprobe_errors", "target", "target_error", "memory",
              "vmstat", "memory_psi",
              "thp_effective", "effective_readback", "gpu", "runtime_witnesses", "error",
              "probe_over_budget"}
    row = _closed(value, fields, "lifecycle sample")
    if (row["schema"] != SAMPLE_SCHEMA or row["phase"] not in PHASES
            or row["status"] not in {"observed", "unknown"}
            or row["kind"] not in {
                "boundary", "periodic", "target_attached", "checkpoint"}):
        raise ObservationError("lifecycle sample state is unsupported")
    for key in ("marker_monotonic_s", "read_started_monotonic_s",
                "read_ended_monotonic_s", "queue_delay_s", "read_duration_s"):
        _finite(row[key], f"sample {key}")
    for key in ("marker_at", "read_started_at", "read_ended_at"):
        _utc(row[key], f"sample {key}")
    for process in row["processes"]:
        _validate_process_row(process)
    for gap in row["census_gaps"]:
        _closed(gap, {"pid", "error"}, "process census gap")
    for error in row["subprobe_errors"]:
        _closed(error, {"probe", "error"}, "subprobe error")
    if row["target"] is not None:
        _validate_process_row(row["target"], target=True)
    if row["memory"] is not None:
        _closed(row["memory"], {"MemAvailable", "SwapFree", "SwapTotal"}, "memory")
    if row["vmstat"] is not None:
        _closed(row["vmstat"], {"pswpin", "pswpout"}, "vmstat")
    if row["memory_psi"] is not None:
        pressure = _closed(row["memory_psi"], {"some", "full"}, "memory PSI")
        for item in pressure.values():
            _closed(item, {"avg10", "avg60", "avg300", "total"}, "memory PSI row")
    if row["thp_effective"] is not None:
        _closed(row["thp_effective"], {"raw", "selected"}, "THP effective state")
    if row["gpu"] is not None:
        gpu = _closed(row["gpu"], {"global_vram_bytes", "target_attribution"}, "GPU state")
        _closed(gpu["target_attribution"],
                {"status", "pid", "start_ticks", "allocated_bytes", "evidence_ref",
                 "reason"},
                "target GPU attribution")
    if row["effective_readback"] is not None:
        _closed(row["effective_readback"],
                {"logical_cpus", "numa_nodes", "thp_mode", "logical_cpus_match",
                 "numa_nodes_match", "thp_mode_match", "recipe_identity_digest",
                 "instrument_identity_digest"}, "effective readback")
    for runtime in row["runtime_witnesses"]:
        runtime = _closed(runtime, {"key", "status", "evidence_ref", "reason"},
                          "runtime witness")
        if runtime["status"] not in RUNTIME_WITNESS_STATES:
            raise ObservationError("runtime witness state is unsupported")


def _validate_interval(value: Any) -> None:
    row = _closed(value, {"start_monotonic_s", "end_monotonic_s", "start_at", "end_at",
                          "elapsed_s", "phase", "status", "census_changes",
                          "potential_foreign_overlap", "memory_delta"},
                  "observation interval")
    if row["phase"] not in PHASES or row["status"] not in {"observed", "unknown"}:
        raise ObservationError("observation interval state is unsupported")
    census = _closed(row["census_changes"], {"born", "disappeared"}, "census changes")
    for identity in [*census["born"], *census["disappeared"]]:
        _closed(identity, {"pid", "start_ticks"}, "process census identity")
    for overlap in row["potential_foreign_overlap"]:
        overlap = _closed(overlap, {"pid", "start_ticks", "process_total_cpu_tick_delta",
                                    "potential_physical_claim_overlap", "verification"},
                          "potential foreign overlap")
        _closed(overlap["verification"], {"status", "kind", "evidence_ref", "reason"},
                "foreign overlap verification")
    if row["memory_delta"] is not None:
        _closed(row["memory_delta"], {"mem_available_kb", "swap_free_kb", "pswpin",
                                      "pswpout", "psi_some_total", "psi_full_total"},
                "memory interval delta")


def validate_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"schema", "detector_version", "observation_id", "backend",
              "instrument_identity_digest", "recipe_identity_digest", "clock_domain",
              "cadence_s", "gap_limit_s", "started_monotonic_s", "ended_monotonic_s",
              "started_at", "ended_at", "boot_id", "worker_binding", "held_claim",
              "requested_effective_state", "budgets", "topology", "target_binding",
              "phase_boundaries", "load_window", "samples", "intervals", "observer_cost",
              "shutdown", "completeness", "issues", "residency", "verdict",
              "content_sha256"}
    row = dict(_closed(value, fields, "lifecycle observation"))
    if row["schema"] != OBSERVATION_SCHEMA or row["detector_version"] != DETECTOR_VERSION:
        raise ObservationError("observation schema/detector is unsupported")
    supplied = _sha(row.pop("content_sha256"), "observation content digest")
    if supplied != _digest(row):
        raise ObservationError("observation content digest mismatch")
    _validate_worker(row["worker_binding"])
    _closed(row["held_claim"], {"logical_cpus", "gpu_devices", "physical_cpus"},
            "held claim")
    _closed(row["requested_effective_state"], {"logical_cpus", "numa_nodes", "thp_mode"},
            "requested effective state")
    _closed(row["budgets"], BUDGET_FIELDS, "observation budgets")
    if row["topology"] is not None:
        topology = dict(_closed(row["topology"], {
            "schema", "logical_cpus", "physical_sibling_groups", "cpu_to_physical_group",
            "cpu_to_numa_node", "topology_digest"}, "topology"))
        topology_digest = topology.pop("topology_digest")
        if _sha(topology_digest, "topology digest") != _digest(topology):
            raise ObservationError("topology digest mismatch")
    if row["target_binding"] is not None:
        binding = _closed(row["target_binding"],
                          {"pid", "start_ticks", "boot_id", "worker_binding", "binding_ref"},
                          "target binding")
        _validate_worker(binding["worker_binding"])
    for boundary in row["phase_boundaries"]:
        boundary = _closed(boundary, {"phase", "monotonic_s", "captured_at"},
                           "phase boundary")
        if boundary["phase"] not in PHASES:
            raise ObservationError("phase boundary is unsupported")
    load_window = _closed(row["load_window"], {"start", "end", "status"}, "load window")
    if load_window["status"] not in {"observed", "unknown"}:
        raise ObservationError("load window state is unsupported")
    _closed(row["shutdown"], {"status", "successor_permitted", "late_writes_accepted"},
            "shutdown")
    if row["shutdown"]["late_writes_accepted"] is not False:
        raise ObservationError("observation cannot accept late writes")
    if ((row["shutdown"]["status"] == "resolved")
            != (row["shutdown"]["successor_permitted"] is True)):
        raise ObservationError("shutdown status disagrees with successor fence")
    verdict = _closed(row["verdict"], {"status", "reason"}, "verdict")
    if verdict["status"] != "not_evaluated":
        raise ObservationError("observation cannot carry a policy verdict")
    for sample in row["samples"]:
        _validate_sample(sample)
    for interval in row["intervals"]:
        _validate_interval(interval)
    observer_cost = _closed(row["observer_cost"], {
        "reader_seconds", "reader_cost_status", "completed_read_count",
        "failed_read_count", "oversized_sample_count", "unresolved_read_count",
        "observation_seconds", "fraction", "sample_count", "sample_budget",
        "dropped_sample_count", "dropped_marker_count", "retained_bytes",
        "retained_byte_budget"}, "observer cost")
    if observer_cost["reader_cost_status"] not in {"complete", "unknown"}:
        raise ObservationError("observer reader cost state is unsupported")
    for key in ("completed_read_count", "failed_read_count", "oversized_sample_count",
                "unresolved_read_count"):
        _integer(observer_cost[key], f"observer cost {key}")
    if (observer_cost["reader_cost_status"] == "complete"
            and observer_cost["unresolved_read_count"] != 0):
        raise ObservationError("complete observer reader cost has unresolved active reads")
    if row["completeness"] not in {"complete", "unknown"}:
        raise ObservationError("observation completeness is unsupported")
    for issue in row["issues"]:
        _closed(issue, {"code", "detail", "counts"}, "observation issue")
    residency = _closed(row["residency"],
                        {"status", "measurement_samples", "positive_samples", "reason"},
                        "residency")
    if residency["status"] not in {"observed", "not_observed", "unknown",
                                    "not_applicable"}:
        raise ObservationError("residency state is unsupported")
    if (row["completeness"] == "complete") != (not row["issues"]):
        raise ObservationError("observation completeness disagrees with diagnostics")
    row["content_sha256"] = supplied
    return _json_copy(row)


__all__ = ["ARTIFACT_SCHEMA", "CONTEXT_SCHEMA", "DETECTOR_VERSION", "FilesystemProbe",
           "FOREIGN_KINDS", "INSTRUMENT_SCHEMA", "OBSERVATION_SCHEMA", "ObservationError",
           "ObservationSession", "ObserverShutdownUnresolved", "PHASES",
           "RUNTIME_WITNESS_STATES", "SAMPLE_SCHEMA",
           "callable_identity", "loaded_instrument_identity", "parse_cpu_list",
           "parse_proc_stat", "physical_footprint", "prepare_artifact_identity",
           "read_topology", "validate_context", "validate_instrument_identity",
           "validate_observation"]
