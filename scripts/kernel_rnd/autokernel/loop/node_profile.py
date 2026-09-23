"""Out-of-band instrumented node/host/engram profiling of the loop's own anchor.

The perf capture in `cpu_profile` samples the MEASURED binary and answers "which
symbols burned user cycles". It cannot answer "which ggml OP owns the wall", "what did
the host spend outside ggml" or "how often did the Engram row gather fault", because
those three numbers exist only in counters compiled in behind `-DGGML_CPU_PROF=ON` --
and the measured binary must stay uninstrumented.

So this is a SECOND, out-of-band capture inside the same reprofile stage. It builds a
profiling SIBLING of the current anchor (same source tree, same commit, the same CPU
recipe plus `GGML_CPU_PROF=ON`) into `<anchor_build>-prof`, launches it ONLY inside the
profile window with the instruments' env gates and dump paths set, replays the SAME
frozen requests the perf capture replays, waits for the server to exit so the at-exit
JSON dumps land, and parses the three dumps.

What it produces is a SHARE view, never a number. Instrumentation moves absolute time;
it does not reorder a 28%-of-wall op behind a 3% one. So per-op shares transfer to the
uninstrumented binary and absolutes do not -- and nothing here is ever a baseline, an
arm, or compared against a measured number.

A missing, unreadable or malformed dump is recorded as `absent` with its reason. It
never fails the profile stage and never fabricates a value: the perf capture stands on
its own, and a null from an instrument that is not in the binary is not evidence.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

NODE_PROFILE_SCHEMA = "epyc.autokernel.loop_node_profile.v1"
NODE_DUMP_SCHEMA = "ggml-cpu-prof/1"
HOST_DUMP_SCHEMA = "llama-host-prof/1"
ENGRAM_DUMP_SCHEMA = "dsv41-engram-profile/1"
#: Suffix of the profiling sibling build directory beside the measured anchor build.
PROFILING_BUILD_SUFFIX = "-prof"
#: Compile-time gate. The two ggml/llama instruments share it; the engram counters are
#: compiled in unconditionally and gated only at run time.
PROFILING_DEFINE = ("GGML_CPU_PROF", "ON")

CPU_PROF_GATE_ENV = "GGML_CPU_PROF"
CPU_PROF_JSON_ENV = "GGML_CPU_PROF_JSON_FILE"
HOST_PROF_JSON_ENV = "LLAMA_HOST_PROF_JSON_FILE"
ENGRAM_JSON_ENV = "LLAMA_ENGRAM_PROF_JSON_FILE"
ENGRAM_LEVEL_ENV = "LLAMA_ENGRAM_PROFILE_LEVEL"
#: The writer's legacy alias for `ENGRAM_JSON_ENV`, read only when the canonical one is
#: unset. Never SET here -- one path, one spelling -- but refused on a measured arm.
ENGRAM_LEGACY_PATH_ENV = "LLAMA_ENGRAM_PROFILE"
#: Every variable this instrument sets or that turns one of its dumps on. A MEASURED arm
#: carrying any of them is refused: the profiled and the measured launch are never one.
PROFILE_ENV_KEYS = frozenset({CPU_PROF_GATE_ENV, CPU_PROF_JSON_ENV, HOST_PROF_JSON_ENV,
                              ENGRAM_JSON_ENV, ENGRAM_LEVEL_ENV, ENGRAM_LEGACY_PATH_ENV})
#: (section, env variable naming the file, file name under the profile directory).
DUMPS = (("node", CPU_PROF_JSON_ENV, "ggml-cpu-prof.json"),
         ("host", HOST_PROF_JSON_ENV, "llama-host-prof.json"),
         ("engram", ENGRAM_JSON_ENV, "dsv41-engram-prof.json"))
ENGRAM_LEVELS = (1, 2)
MAX_DUMP_BYTES = 64 * 1024 * 1024
MAX_ROWS = 16384
LIMITATIONS = (
    "measured on an INSTRUMENTED sibling build, never on the measured binary",
    "per-op and per-phase SHARES transfer between the profiled and unprofiled builds; "
    "absolute times do not",
    "never a baseline, never an arm, never compared against a measured number",
    "node identity is the graph node INDEX, stable only within one graph shape",
    "wall includes this node's barrier wait and straggler imbalance as seen by thread 0",
    "prefill and decode are never pooled; profiled prefill timings are inadmissible",
    "engram fault counts are meaningless without fault_source and op_level >= 2; below "
    "that a zero is UNMEASURED, not an absence of faults",
    "the three dumps share no run id, only time: node values are per accumulated GRAPH "
    "EVAL, host decode values per token (and raw totals when n_eval is 0), engram values "
    "raw run totals -- normalise before dividing one by another",
    "host phase rows are two overlapping families: the three ctx.* rows already contain "
    "the per-input-class rows and the two are never summed")
BASIS = "instrumented sibling build of the current anchor, same commit and frozen requests"


class NodeProfileRefused(ValueError):
    """A dump is absent or does not carry the fields this reader needs."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NodeProfileRefused(f"{label} must be an object")
    return dict(value)


def _rows(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise NodeProfileRefused(f"{label} must be an array")
    if len(value) > MAX_ROWS:
        raise NodeProfileRefused(f"{label} exceeds the supported row bound")
    return [_object(item, f"{label}[]") for item in value]


def _number(row: Mapping[str, Any], key: str, label: str, *, default=None) -> float:
    value = row.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NodeProfileRefused(f"{label}.{key} must be a number")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        raise NodeProfileRefused(f"{label}.{key} must be finite and non-negative")
    return value


def _text(row: Mapping[str, Any], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise NodeProfileRefused(f"{label}.{key} must be a non-empty string")
    return value


def profiling_build_dir(anchor_build: Path | str) -> Path:
    """The sibling directory beside the measured anchor build, never inside it."""
    build = Path(anchor_build)
    return build.parent / (build.name + PROFILING_BUILD_SUFFIX)


def launch_env(directory: Path | str, *, level: int = 1) -> dict[str, str]:
    """The exact environment the profiling sibling is launched with.

    Each dump has its own path variable and none of them defaults to a path, so an
    instrumented binary launched without these writes nothing. `GGML_CPU_PROF` is the
    runtime gate for dumps (a) and (b); the engram dump needs only its own path, and
    `LLAMA_ENGRAM_PROFILE_LEVEL=2` is what raises the op half to fault attribution.
    """
    if level not in ENGRAM_LEVELS:
        raise NodeProfileRefused(f"engram profile level must be one of {ENGRAM_LEVELS}")
    paths = dump_paths(directory)
    return {CPU_PROF_GATE_ENV: "1",
            CPU_PROF_JSON_ENV: paths["node"],
            HOST_PROF_JSON_ENV: paths["host"],
            ENGRAM_JSON_ENV: paths["engram"],
            ENGRAM_LEVEL_ENV: str(level)}


def dump_paths(directory: Path | str) -> dict[str, str]:
    root = Path(directory)
    return {name: str(root / filename) for name, _, filename in DUMPS}


def refuse_instrumented_measurement(resolved) -> None:
    """A measured arm may never carry the instrument's env or its build.

    Called on the arm the A/B actually measures, not on the sibling. The check is on
    the launch environment because that is the thing this module writes; the build
    define is separately unreachable from the measured recipe, which never names it.
    """
    present = sorted(PROFILE_ENV_KEYS.intersection(dict(resolved.launch_env)))
    if present:
        raise NodeProfileRefused(
            f"measured launch carries instrumentation environment {present}: the "
            f"profiled sibling and the measured arm must never be the same launch")


def parse_node_dump(raw: bytes | str) -> dict[str, Any]:
    """`ggml_cpu_prof_write_json`'s object: per-op and per-weight-path wall shares."""
    body = _object(_load(raw, "node dump"), "node dump")
    if body.get("schema") != NODE_DUMP_SCHEMA:
        raise NodeProfileRefused(f"node dump schema is not {NODE_DUMP_SCHEMA}")
    total_wall = _number(body, "total_wall_us", "node dump")
    if total_wall <= 0:
        raise NodeProfileRefused("node dump total_wall_us is not positive")
    ops = []
    for row in _rows(body.get("ops"), "node dump.ops"):
        wall = _number(row, "wall_us", "node dump.ops")
        ops.append({"op": _text(row, "op", "node dump.ops"),
                    "count": _number(row, "count", "node dump.ops"),
                    "compute_us": _number(row, "compute_us", "node dump.ops"),
                    "wall_us": wall, "wall_fraction": wall / total_wall})
    if not ops:
        raise NodeProfileRefused("node dump carries no op rows")
    paths = []
    for row in _rows(body.get("paths", []), "node dump.paths"):
        wall = _number(row, "wall_us", "node dump.paths")
        paths.append({"path": _text(row, "path", "node dump.paths"),
                      "calls": _number(row, "calls", "node dump.paths"),
                      "compute_us": _number(row, "compute_us", "node dump.paths"),
                      "wall_us": wall, "wall_fraction": wall / total_wall,
                      "bytes": _number(row, "bytes", "node dump.paths")})
    ops.sort(key=lambda row: (-row["wall_us"], row["op"]))
    paths.sort(key=lambda row: (-row["wall_us"], row["path"]))
    return {"schema": NODE_DUMP_SCHEMA,
            "graph_evals_accumulated": _number(body, "graph_evals_accumulated", "node dump"),
            "n_nodes": _number(body, "n_nodes", "node dump"),
            "n_threads": _number(body, "n_threads", "node dump"),
            "total_wall_us": total_wall,
            "total_compute_us": _number(body, "total_compute_us", "node dump"),
            "barriers": _number(body, "barriers", "node dump", default=0.0),
            "ops": ops, "weight_paths": paths,
            # The shares are per-op fractions of the SAME denominator, so their sum is
            # a coverage reading of the instrument, not a normalisation to apply.
            "op_wall_fraction_sum": sum(row["wall_fraction"] for row in ops)}


def parse_host_dump(raw: bytes | str) -> dict[str, Any]:
    """`host_prof_write`'s object: the host phases outside the ggml graph.

    Two traps, both from the writer. `n_eval` is filled only if `llama_perf_context()`
    ran, and when it is 0 the `*_us_per_token` keys carry RAW TOTALS wearing a per-token
    name -- so a zero denominator is "unavailable", never "zero tokens". And the phase
    array holds two families at once: the three `ctx.*` rows are a superset that already
    CONTAINS the per-input-class rows, so the two must never be summed.
    """
    body = _object(_load(raw, "host dump"), "host dump")
    if body.get("schema") != HOST_DUMP_SCHEMA:
        raise NodeProfileRefused(f"host dump schema is not {HOST_DUMP_SCHEMA}")
    decode = _number(body, "decode_us_per_token", "host dump")
    n_eval = _number(body, "n_eval", "host dump")
    usable = n_eval > 0 and decode > 0
    phases = []
    for row in _rows(body.get("phases"), "host dump.phases"):
        per_token = _number(row, "decode_us_per_token", "host dump.phases")
        name = _text(row, "phase", "host dump.phases")
        phases.append({"phase": name,
                       "family": "context_phase" if name.startswith("ctx.")
                                 else "graph_input_class",
                       "decode_us_per_token": per_token,
                       "decode_calls": _number(row, "decode_calls", "host dump.phases"),
                       "prefill_us_total": _number(row, "prefill_us_total", "host dump.phases"),
                       "prefill_calls": _number(row, "prefill_calls", "host dump.phases"),
                       "decode_fraction": (per_token / decode) if usable else None})
    phases.sort(key=lambda row: (-row["decode_us_per_token"], row["phase"]))
    return {"schema": HOST_DUMP_SCHEMA, "n_eval": n_eval,
            "n_p_eval": _number(body, "n_p_eval", "host dump"),
            "decode_us_per_token": decode,
            "prefill_us_per_token": _number(body, "prefill_us_per_token", "host dump"),
            "decode_denominator": "n_eval" if usable else "unavailable",
            "phase_families_overlap": True, "phases": phases}


def parse_engram_dump(raw: bytes | str) -> dict[str, Any]:
    """The DS41 engram artifact: gather cost and the fault mix behind it."""
    body = _object(_load(raw, "engram dump"), "engram dump")
    if body.get("schema") != ENGRAM_DUMP_SCHEMA:
        raise NodeProfileRefused(f"engram dump schema is not {ENGRAM_DUMP_SCHEMA}")
    host = _object(body.get("host"), "engram dump.host")
    layers = []
    for index, row in enumerate(_rows(body.get("layers", []), "engram dump.layers")):
        step = _object(row.get("decode"), f"engram dump.layers[{index}].decode")
        rows_read = _number(step, "rows", "engram decode")
        tokens = _number(step, "n_tokens", "engram decode")
        unique = _number(step, "rows_uniq_in_token", "engram decode")
        hits = _rows_of_numbers(step.get("cache_hit", []), "engram decode.cache_hit")
        layers.append({
            "index": int(_number(row, "index", "engram dump.layers", default=index)),
            "table_rows": _number(row, "table_rows", "engram dump.layers"),
            "n_tokens": tokens, "n_gated": _number(step, "n_gated", "engram decode"),
            "rows": rows_read, "rows_uniq_in_token": unique,
            "rows_same_as_prev": _number(step, "rows_same_as_prev", "engram decode"),
            # The artifact carries counts on purpose; the ratios are the loop's to
            # derive and to label, so they are computed here and never stored there.
            "intra_token_duplication": (1.0 - step_ratio(unique, rows_read))
            if rows_read > 0 else None,
            "cache_hit_rate": [step_ratio(value, rows_read) for value in hits]
            if rows_read > 0 else None,
            "cache_capacity_rows": _rows_of_numbers(
                row.get("cache_capacity_rows", []), "engram dump.layers.cache_capacity_rows")})
    tables = []
    op = _object(body.get("op", {}), "engram dump.op")
    for index, row in enumerate(_rows(op.get("tables", []), "engram dump.op.tables")):
        step = _object(row.get("decode"), f"engram dump.op.tables[{index}].decode")
        tables.append({
            "slot": int(_number(row, "slot", "engram dump.op.tables", default=index)),
            # -1 is the writer's "this table matched no engram layer"; it is a real
            # state, not a missing value, so it is kept rather than normalised away.
            "engram_layer": int(row.get("engram_layer", -1))
            if isinstance(row.get("engram_layer", -1), int) else -1,
            "n_calls": _number(step, "n_calls", "engram op decode"),
            "n_rows": _number(step, "n_rows", "engram op decode"),
            "us_span_ith0": _number(step, "us_span_ith0", "engram op decode"),
            "us_cpu": _number(step, "us_cpu", "engram op decode"),
            "n_thread_spans": _number(step, "n_thread_spans", "engram op decode"),
            "minflt": _number(step, "minflt", "engram op decode", default=0.0),
            "majflt": _number(step, "majflt", "engram op decode", default=0.0)})
    level = int(_number(body, "level", "engram dump", default=1))
    op_level = int(_number(body, "op_level", "engram dump", default=0))
    resolved = body.get("op_profiler") == "resolved"
    span = sum(row["us_span_ith0"] for row in tables)
    decode_span = _number(host, "us_decode_step_span", "engram dump.host", default=0.0)
    tokens = max((row["n_tokens"] for row in layers), default=0.0)
    fault_source = body.get("fault_source")
    measured_faults = resolved and op_level >= 2 and fault_source == "getrusage_thread"
    return {"schema": ENGRAM_DUMP_SCHEMA, "level": level, "op_level": op_level,
            "op_profiler": body.get("op_profiler"),
            # With the op half unresolved the whole `op` object is zeros, and those
            # zeros are not measurements of anything.
            "op_counts_are_measured": resolved,
            # A zero fault count below level 2 is UNMEASURED. Never let it read as "no
            # faults": the whole residency verdict hangs on which of the two it is.
            "fault_source": fault_source if isinstance(fault_source, str) else "none",
            "fault_counts_are_measured": measured_faults,
            "n_engram": _number(body, "n_engram", "engram dump", default=float(len(layers))),
            "n_col": _number(body, "n_col", "engram dump", default=0.0),
            "host": {"us_hash": _number(host, "us_hash", "engram dump.host", default=0.0),
                     "us_observe": _number(host, "us_observe", "engram dump.host", default=0.0),
                     "us_decode_step_span": decode_span,
                     "n_decode_step_span": _number(host, "n_decode_step_span",
                                                   "engram dump.host", default=0.0)},
            "layers": layers, "tables": tables,
            # Every derived rate is None unless its own inputs were measured. A
            # fault rate computed from unmeasured zeros would read exactly like a
            # resident table, which is the one reading this instrument exists to decide.
            "gather_share_of_decode": step_ratio(span, decode_span)
            if resolved and decode_span > 0 else None,
            "minflt_per_decode_token": step_ratio(
                sum(row["minflt"] for row in tables), tokens)
            if measured_faults and tokens > 0 else None,
            "majflt_per_decode_token": step_ratio(
                sum(row["majflt"] for row in tables), tokens)
            if measured_faults and tokens > 0 else None}


def step_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise NodeProfileRefused("ratio denominator must be positive")
    return numerator / denominator


def _rows_of_numbers(value: Any, label: str) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise NodeProfileRefused(f"{label} must be an array")
    if len(value) > MAX_ROWS:
        raise NodeProfileRefused(f"{label} exceeds the supported row bound")
    return [_number({"v": item}, "v", label) for item in value]


def _load(raw: bytes | str, label: str) -> Any:
    if isinstance(raw, bytes):
        if len(raw) > MAX_DUMP_BYTES:
            raise NodeProfileRefused(f"{label} exceeds the supported byte bound")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NodeProfileRefused(f"{label} is not UTF-8") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NodeProfileRefused(f"{label} is not valid JSON") from exc


#: The node profiler's op vocabulary, projected onto the SAME family names
#: `cpu_profile._mechanism_family` gives the planner, so the two views of one anchor
#: name one mechanism the same way instead of two.
_OP_FAMILIES = (("MUL_MAT_ID", "moe-expert-matmul"),
                ("GATHER_ROWS", "engram-row-gather"),
                ("FLASH_ATTN", "flash-attention"),
                ("RMS_NORM", "rms-normalization"),
                ("MUL_MAT", "dense-matmul"))


def op_mechanism_family(op: str) -> str:
    upper = str(op).upper()
    for marker, family in _OP_FAMILIES:
        if marker in upper:
            return family
    return "op:" + str(op)


def mechanism_shares(node, *, limit: int = 8) -> list[dict[str, Any]]:
    """Group the per-op wall shares the way `ranked_levers` groups sampled symbols."""
    groups: dict[str, dict[str, Any]] = {}
    for row in node["ops"]:
        family = op_mechanism_family(row["op"])
        group = groups.setdefault(family, {
            "family": family, "wall_us": 0.0, "ops": [],
            "evidence_kind": "instrumented-build per-op wall share"})
        group["wall_us"] += row["wall_us"]
        if len(group["ops"]) < 4:
            group["ops"].append({"op": row["op"], "wall_us": row["wall_us"],
                                 "wall_fraction": row["wall_fraction"]})
    ranked = sorted(groups.values(), key=lambda row: (-row["wall_us"], row["family"]))
    for row in ranked:
        row["wall_fraction"] = row["wall_us"] / node["total_wall_us"]
    return ranked[:limit]


def absent(reason: str) -> dict[str, Any]:
    return {"status": "absent", "reason": str(reason)[:1024]}


def read_dumps(directory: Path | str) -> dict[str, Any]:
    """Parse whatever landed. One unreadable dump never costs the other two."""
    parsers = {"node": parse_node_dump, "host": parse_host_dump,
               "engram": parse_engram_dump}
    paths = dump_paths(directory)
    result = {}
    for name in ("node", "host", "engram"):
        path = Path(paths[name])
        try:
            raw = path.read_bytes()
        except OSError as exc:
            result[name] = absent(f"dump not written: {type(exc).__name__}: {exc}")
            continue
        try:
            result[name] = {"status": "observed", "path": str(path), **parsers[name](raw)}
        except NodeProfileRefused as exc:
            result[name] = absent(str(exc))
    return result


def section(dumps: Mapping[str, Any], *, build: Mapping[str, Any],
            teardown: str | None = None) -> dict[str, Any]:
    """Assemble the planner-facing record; `absent` parts stay absent, never zeroed."""
    node = dumps.get("node", absent("not collected"))
    host = dumps.get("host", absent("not collected"))
    engram = dumps.get("engram", absent("not collected"))
    body = {"schema": NODE_PROFILE_SCHEMA,
            "status": "observed" if node.get("status") == "observed" else "absent",
            "basis": BASIS, "limitations": list(LIMITATIONS),
            "build": dict(build), "teardown": teardown,
            "node": node, "host": host, "engram": engram}
    if node.get("status") == "observed":
        body["ranked_op_shares"] = node["ops"][:12]
        body["weight_path_shares"] = node["weight_paths"]
        body["mechanism_shares"] = mechanism_shares(node)
    else:
        body["reason"] = node.get("reason", "node dump absent")
    if host.get("status") == "observed":
        body["host_phase_shares"] = host["phases"][:12]
    if engram.get("status") == "observed":
        body["engram_fault_mix"] = {
            "level": engram["level"], "op_level": engram["op_level"],
            "fault_source": engram["fault_source"],
            "op_counts_are_measured": engram["op_counts_are_measured"],
            "fault_counts_are_measured": engram["fault_counts_are_measured"],
            "gather_share_of_decode": engram["gather_share_of_decode"],
            "minflt_per_decode_token": engram["minflt_per_decode_token"],
            "majflt_per_decode_token": engram["majflt_per_decode_token"]}
    return body


def cache_key(*, anchor_commit: str, execution_digest: str,
              prompt_manifest_digest: str, scope: str, level: int) -> dict[str, Any]:
    """The perf capture's key, plus the instrument level that changes what is counted."""
    return {"anchor_commit": str(anchor_commit), "execution_digest": str(execution_digest),
            "prompt_manifest_digest": str(prompt_manifest_digest), "scope": str(scope),
            "level": int(level)}


def _cache_path(store_root: Path | str, key: Mapping[str, Any]) -> Path:
    return Path(store_root) / "node-profiles" / (_digest(dict(key)) + ".json")


def cached_observation(*, store_root: Path | str, **key) -> dict[str, Any] | None:
    """Reuse only the same anchor commit, launch, requests, scope and level."""
    reference = cache_key(**key)
    path = _cache_path(store_root, reference)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(body, Mapping) or body.get("cache_key") != reference:
        return None
    observed = body.get("section")
    return dict(observed) if isinstance(observed, Mapping) else None


def retain_observation(observed: Mapping[str, Any], *, store_root: Path | str,
                       **key) -> Path:
    reference = cache_key(**key)
    path = _cache_path(store_root, reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": NODE_PROFILE_SCHEMA, "cache_key": reference,
                                "section": dict(observed)}, sort_keys=True, indent=2),
                    encoding="utf-8")
    return path


def profile_loop(arm_for_env, prompts, *, store_root: Path | str, build: Mapping[str, Any],
                 level: int = 1, timeout_s: int = 1800, launch=None) -> dict[str, Any]:
    """Launch the instrumented sibling once, replay the frozen requests, read the dumps.

    `arm_for_env` takes the instrument environment and returns the resolved launch for
    the SIBLING build -- the caller owns the rebind, exactly as the perf capture lets
    `run.reprofile` own `_cpu_arm`. Nothing here touches the measured arm.
    """
    directory = Path(store_root) / "node-profiles" / ("run-" + _digest(
        {"build": dict(build), "level": int(level)})[:32])
    directory.mkdir(parents=True, exist_ok=True)
    for name in dump_paths(directory).values():
        Path(name).unlink(missing_ok=True)
    environment = launch_env(directory, level=level)
    arm = arm_for_env(environment)
    present = {key: value for key, value in dict(arm.launch_env).items()
               if key in PROFILE_ENV_KEYS}
    if present != environment:
        raise NodeProfileRefused("profiling launch does not carry the instrument environment")
    frozen = prompts.requests(tuple(item.prompt_id for item in prompts.prompts), arm.template)
    if launch is None:
        from . import serving
        launch = serving._measure_once
    observation: list = []
    # The dumps are written by atexit handlers, so the server must EXIT, not be killed.
    # `_measure_once` terminates and waits; a launch that had to be killed is recorded
    # as such and its dumps simply do not exist.
    launch(arm.template, Path(arm.build_dir), arm.port, boot_timeout_s=timeout_s,
           resolved_recipe=arm, frozen_requests=frozen, observation=observation)
    teardown = observation[-1].get("teardown") if observation else None
    return section(read_dumps(directory), build=build, teardown=teardown)


__all__ = ["BASIS", "CPU_PROF_GATE_ENV", "CPU_PROF_JSON_ENV", "DUMPS",
           "ENGRAM_JSON_ENV", "ENGRAM_LEGACY_PATH_ENV", "ENGRAM_LEVEL_ENV",
           "HOST_PROF_JSON_ENV", "LIMITATIONS", "NODE_PROFILE_SCHEMA",
           "NodeProfileRefused", "PROFILE_ENV_KEYS", "PROFILING_BUILD_SUFFIX",
           "PROFILING_DEFINE", "absent", "cache_key", "cached_observation",
           "dump_paths", "launch_env", "mechanism_shares", "op_mechanism_family",
           "parse_engram_dump", "parse_host_dump", "parse_node_dump", "profile_loop",
           "profiling_build_dir", "read_dumps", "refuse_instrumented_measurement",
           "retain_observation", "section"]
