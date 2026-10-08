"""Prospective SC55 capture carrier; never launches a producer or assigns a grade."""
from __future__ import annotations
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "epyc.graph_profile_capture.v2"
KNOBS = ("GGML_CPU_PROF", "GGML_CPU_PROF_PERNODE_FILE", "GGML_CPU_PROF_THREADS",
         "GGML_CPU_PROF_NODES", "GGML_CPU_PROF_NODES_FILE", "GGML_CPU_PROF_MM",
         "GGML_CPU_PROF_SKIP", "GGML_CPU_PROF_NNODES_EQ", "GGML_CPU_PROF_NNODES_MIN",
         "GGML_CPU_PROF_NNODES_MAX", "GGML_CPU_PROF_SPIKE_US", "GGML_IQK")
ROLES = {"pernode", "node_table", "log", "run_receipt", "graph_identity", "compiled_strings"}

class Refusal(ValueError):
    """Required contemporaneous evidence is absent or ambiguous."""

def require(ok, reason):
    if not ok:
        raise Refusal(reason)

def exact(obj, keys):
    require(type(obj) is dict and set(obj) == set(keys), "unexpected or missing fields")

def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "invalid integer")

def stamp(value):
    require(type(value) is str, "missing timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Refusal("invalid timestamp") from exc
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "timestamp must be UTC")
    return result

def writer_utc():
    """Actual writer clock; synthetic controls may mock this pure clock seam."""
    return datetime.now(timezone.utc).isoformat()

def digest(value, width=64):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{%d}" % width, value) is not None, "invalid identity digest")

def receipt(path, required_strings=()):
    p = Path(path)
    require(p.is_absolute() and not p.is_symlink(), "absolute non-symlink file required")
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        import stat
        require(stat.S_ISREG(before.st_mode), "regular file required")
        h = hashlib.sha256()
        needles = {value.encode() for value in required_strings}
        found = set()
        tail = b""
        overlap = max((len(value) for value in needles), default=1) - 1
        with os.fdopen(fd, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
                scan = tail + block
                found.update(value for value in needles if value in scan)
                tail = scan[-overlap:] if overlap else b""
        after = os.fstat(fd)
        require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns), "file changed while sealing")
        current = os.stat(p, follow_symlinks=False)
        require((current.st_dev, current.st_ino) == (after.st_dev, after.st_ino), "path replaced")
        require(found == needles, "compiled string missing from bound bytes")
        return {"path": str(p), "size": after.st_size, "sha256": h.hexdigest(),
                "device": after.st_dev, "inode": after.st_ino, "mtime_ns": after.st_mtime_ns}
    finally:
        os.close(fd)

def write_new(path, data):
    encoded = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    # Exclusive creation prevents silently overwriting any prior capture phase.
    with open(path, "xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt(Path(path).absolute())

def load(path):
    def pairs(items):
        obj = {}
        for key, value in items:
            require(key not in obj, "duplicate JSON field")
            obj[key] = value
        return obj
    with open(path, encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs,
                         parse_constant=lambda value: require(False, "nonfinite JSON"))

def validate_measurement_metadata(metadata):
    """Required owner-authored native labels; missing gradable elements stay explicit null."""
    exact(metadata, {"schema", "date", "category", "protocol_id", "metrics"})
    require(metadata["schema"] == "epyc.graph_profile_measurement_metadata.v1", "unknown measurement metadata")
    from datetime import date
    require(metadata["date"] is None or type(metadata["date"]) is str, "unknown date type")
    if metadata["date"] is not None:
        try: parsed = date.fromisoformat(metadata["date"])
        except ValueError as exc: raise Refusal("invalid recorded date") from exc
        require(parsed.isoformat() == metadata["date"], "noncanonical recorded date")
    require(metadata["category"] in ("OPTIMUM", "BASELINE", "CANDIDATE"), "missing/unknown category")
    require(metadata["protocol_id"] is None or type(metadata["protocol_id"]) is str and bool(metadata["protocol_id"].strip()), "ambiguous protocol declaration")
    require(type(metadata["metrics"]) is list and metadata["metrics"], "missing measurement identities")
    seen = set()
    node_units = {"compute_us": "us", "wall_us": "us", "wall_max_us": "us", "thr_max_us": "us", "thr_mean_us": "us", "thr_min_us": "us"}
    path_units = {"calls_per_eval": "calls/eval", "compute_ms": "ms", "wall_ms": "ms", "bytes_per_eval": "bytes/eval", "GBs_on_compute": "GB/s", "GBs_on_wall": "GB/s"}
    for metric in metadata["metrics"]:
        exact(metric, {"measurement_id", "metric", "selector", "metric_direction", "unit", "claim", "reps_basis", "attestation_role"})
        for key in ("measurement_id", "metric", "claim"):
            require(type(metric[key]) is str and bool(metric[key].strip()), "missing measurement identity/claim")
        require(metric["measurement_id"] not in seen, "duplicate measurement identity")
        seen.add(metric["measurement_id"])
        require(metric["metric_direction"] in ("higher_better", "lower_better"), "missing/unknown metric direction")
        selector = metric["selector"]
        exact(selector, {"kind", "identity", "field"})
        if selector["kind"] == "node":
            integer(selector["identity"])
            require(selector["field"] in node_units and metric["unit"] == node_units[selector["field"]], "unknown node field/unit")
            require(metric["reps_basis"] == "native_node_evals", "unknown node reps basis")
        else:
            require(selector["kind"] == "path" and selector["identity"] in ("dense_mul_mat", "expert_mul_mat_id", "lm_head"), "unknown path identity")
            require(selector["field"] in path_units and metric["unit"] == path_units[selector["field"]], "unknown path field/unit")
            require(metric["reps_basis"] == "native_accumulated_graph_evals", "unknown path reps basis")
        require(metric["attestation_role"] is None or metric["attestation_role"] == ("pernode" if selector["kind"] == "node" else "log"), "ambiguous native attestation role")

def validate_pre(p):
    exact(p, {"capture_id", "owner", "recorded_at", "source", "binary", "knobs", "threads",
              "graph", "eval", "iqk", "raw_paths", "outputs", "measurement_metadata"})
    validate_measurement_metadata(p["measurement_metadata"])
    for key in ("capture_id", "owner"):
        require(type(p[key]) is str and bool(p[key].strip()), "missing owner/capture identity")
    stamp(p["recorded_at"])
    exact(p["source"], {"commit", "tree", "profiler_source_sha256"})
    digest(p["source"]["commit"], 40); digest(p["source"]["tree"], 40)
    digest(p["source"]["profiler_source_sha256"])
    exact(p["binary"], {"path", "sha256", "mtime_ns", "build_commit", "compiled_knobs", "compiled_strings_sha256"})
    digest(p["binary"]["compiled_strings_sha256"])
    digest(p["binary"]["sha256"]); digest(p["binary"]["build_commit"], 40)
    require(p["binary"]["build_commit"] == p["source"]["commit"], "build/source mismatch")
    integer(p["binary"]["mtime_ns"], 1)
    require(type(p["binary"]["path"]) is str and Path(p["binary"]["path"]).is_absolute(), "binary path required")
    require(type(p["binary"]["compiled_knobs"]) is list and
            all(type(value) is str for value in p["binary"]["compiled_knobs"]) and
            set(p["binary"]["compiled_knobs"]) == set(KNOBS), "compiled knob proof incomplete")
    exact(p["knobs"], KNOBS)
    for value in p["knobs"].values():
        require(value is None or type(value) is str, "knob must record absent or exact string")
    require(p["knobs"]["GGML_CPU_PROF"] is not None and
            p["knobs"]["GGML_CPU_PROF_PERNODE_FILE"] is not None, "profiler/output disabled")
    integer(p["threads"], 1)
    exact(p["graph"], {"identity_sha256", "identity_method", "filter_semantics"})
    digest(p["graph"]["identity_sha256"])
    require(p["graph"]["identity_method"] == "owner_structural_receipt", "node count is not graph identity")
    require(p["graph"]["filter_semantics"] == "native_nnodes_eq_min_max_after_global_skip", "unknown graph filter")
    for key in ("GGML_CPU_PROF_NNODES_EQ", "GGML_CPU_PROF_NNODES_MIN", "GGML_CPU_PROF_NNODES_MAX", "GGML_CPU_PROF_SPIKE_US"):
        value = p["knobs"][key]
        require(value is None or re.fullmatch(r"0|[1-9][0-9]*", value) is not None and int(value) <= 2147483647, "ambiguous native integer gate")
    lo, hi, eq = (p["knobs"]["GGML_CPU_PROF_NNODES_" + key] for key in ("MIN", "MAX", "EQ"))
    require(lo is None or hi is None or int(lo) <= int(hi), "empty graph filter")
    require(eq is None or (lo is None or int(eq) >= int(lo)) and (hi is None or int(eq) <= int(hi)), "empty graph filter")
    exact(p["eval"], {"skip", "warmup_semantics", "accumulation_semantics", "measuring_thread", "dispersion_semantics"})
    integer(p["eval"]["skip"])
    skip = p["knobs"]["GGML_CPU_PROF_SKIP"]
    require(skip is None and p["eval"]["skip"] == 0 or
            type(skip) is str and re.fullmatch(r"0|[1-9][0-9]*", skip) is not None and int(skip) <= 2147483647 and int(skip) == p["eval"]["skip"], "skip gate mismatch")
    require(p["eval"]["measuring_thread"] == "thread0_wall_compute", "unknown timing perspective")
    semantics = {"warmup_semantics": "skip_first_global_graph_evaluations_not_phase_detection",
                 "accumulation_semantics": "post_skip_shape_filtered_node_index_aggregate",
                 "dispersion_semantics": "wall_max_accumulated_eval_and_threshold_spikes_no_thread_vector"}
    for key, known in semantics.items():
        require(p["eval"][key] == known, "unknown eval semantics")
    require(p["knobs"]["GGML_IQK"] in (None, "0", "1"), "unknown IQK gate")
    require(p["iqk"] in ("enabled", "disabled"), "unknown IQK state")
    require((p["knobs"]["GGML_IQK"] == "1") == (p["iqk"] == "enabled"), "IQK inconsistency")
    exact(p["raw_paths"], ROLES)
    required_paths = {role: value for role, value in p["raw_paths"].items() if role != "node_table"}
    paths = list(required_paths.values())
    require(all(type(v) is str and Path(v).is_absolute() for v in paths) and len(set(paths)) == len(paths), "ambiguous raw paths")
    require(p["knobs"]["GGML_CPU_PROF_PERNODE_FILE"] == p["raw_paths"]["pernode"], "output path mismatch")
    exact(p["outputs"], {"node_table", "log", "mm"})
    nodes = p["knobs"]["GGML_CPU_PROF_NODES"] is not None
    table_path = p["knobs"]["GGML_CPU_PROF_NODES_FILE"]
    expected_table = {"availability": "unavailable", "path": None} if not nodes else {
        "availability": "requested_file" if table_path is not None else "requested_stderr",
        "path": table_path if table_path is not None else p["raw_paths"]["log"]}
    require(p["outputs"]["node_table"] == expected_table and
            p["raw_paths"]["node_table"] == expected_table["path"], "node table gate/path mismatch")
    if table_path is not None and nodes:
        require(type(table_path) is str and Path(table_path).is_absolute() and table_path not in paths, "ambiguous table path")
    require(p["outputs"]["log"] == {"availability": "retained_stderr", "path": p["raw_paths"]["log"]}, "missing retained log")
    mm = p["knobs"]["GGML_CPU_PROF_MM"] is not None
    require(p["outputs"]["mm"] == {"availability": "requested_stderr" if mm else "unavailable", "path": p["raw_paths"]["log"] if mm else None}, "MM gate/path mismatch")

def pre_custody(p):
    binary = receipt(p["binary"]["path"], KNOBS)
    graph = receipt(p["raw_paths"]["graph_identity"])
    strings = receipt(p["raw_paths"]["compiled_strings"], KNOBS)
    require(binary["sha256"] == p["binary"]["sha256"] and binary["mtime_ns"] == p["binary"]["mtime_ns"], "binary drift")
    require(graph["sha256"] == p["graph"]["identity_sha256"], "graph receipt drift")
    require(strings["sha256"] == p["binary"]["compiled_strings_sha256"], "compiled proof drift")
    return {"binary": binary, "graph_identity": graph, "compiled_strings": strings}


def begin(path, provenance):
    """Inference owner records pre-run facts; caller must supply observed identities."""
    validate_pre(provenance)
    custody = pre_custody(provenance)
    written_at = writer_utc()
    require(stamp(provenance["recorded_at"]) <= stamp(written_at), "future pre assertion")
    return write_new(path, {"schema": SCHEMA, "phase": "pre", "provenance": provenance,
                            "observed_custody": custody, "writer_observed_at": written_at})

def validate_observation(p, observation):
    exact(observation, {"started_at", "observed_at", "owner", "capture_id", "binary_sha256", "effective_knobs", "effective_threads", "graph_identity_sha256", "thread_availability"})
    integer(observation["effective_threads"], 1)
    date = p["measurement_metadata"]["date"]
    require(date is None or date == stamp(observation["started_at"]).date().isoformat(), "native measurement date/window mismatch")
    require(stamp(p["recorded_at"]) <= stamp(observation["started_at"]) <= stamp(observation["observed_at"]), "noncontemporaneous window")
    for key in ("owner", "capture_id"):
        require(observation[key] == p[key], "capture custody mismatch")
    require(observation["binary_sha256"] == p["binary"]["sha256"] and
            observation["effective_knobs"] == p["knobs"] and observation["effective_threads"] == p["threads"] and
            observation["graph_identity_sha256"] == p["graph"]["identity_sha256"], "during provenance drift")
    available = "aggregate_max_mean_min" if p["knobs"]["GGML_CPU_PROF_THREADS"] is not None else "unavailable"
    require(observation["thread_availability"] == available, "zero threads are unavailable without gate")

def during(path, pre_path, observation):
    pre = load(pre_path)
    exact(pre, {"schema", "phase", "provenance", "observed_custody", "writer_observed_at"})
    require(pre["schema"] == SCHEMA and pre["phase"] == "pre", "wrong pre carrier")
    validate_pre(pre["provenance"])
    validate_observation(pre["provenance"], observation)
    written_at = writer_utc()
    require(stamp(pre["provenance"]["recorded_at"]) <= stamp(pre["writer_observed_at"]) <=
            stamp(observation["started_at"]) <= stamp(observation["observed_at"]) <= stamp(written_at),
            "historical or future during window")
    return write_new(path, {"schema": SCHEMA, "phase": "during", "pre": receipt(Path(pre_path).absolute()),
                            "observation": observation, "writer_observed_at": written_at})

def finalize(path, pre_path, during_path, closed):
    """Seal only after owner proves producer exit and all raw handles closed."""
    pre = load(pre_path)
    exact(pre, {"schema", "phase", "provenance", "observed_custody", "writer_observed_at"})
    require(pre["schema"] == SCHEMA and pre["phase"] == "pre", "wrong pre carrier")
    p = pre["provenance"]
    validate_pre(p)
    d = load(during_path)
    exact(d, {"schema", "phase", "pre", "observation", "writer_observed_at"})
    require(d["schema"] == SCHEMA and d["phase"] == "during" and d["pre"] == receipt(Path(pre_path).absolute()), "pre custody changed")
    validate_observation(p, d["observation"])
    exact(closed, {"ended_at", "closed_at", "producer_exited", "all_handles_closed", "owner", "capture_id"})
    require(closed["producer_exited"] is True and closed["all_handles_closed"] is True, "raw files not closed")
    require(closed["owner"] == p["owner"] and closed["capture_id"] == p["capture_id"], "closure custody mismatch")
    require(stamp(d["observation"]["observed_at"]) <= stamp(closed["ended_at"]) <= stamp(closed["closed_at"]), "invalid closed window")
    observed = pre_custody(p)
    require(observed == pre["observed_custody"], "pre-run custody changed")
    raw = {role: None if raw_path is None else
           observed[role] if role in ("graph_identity", "compiled_strings") else receipt(raw_path)
           for role, raw_path in p["raw_paths"].items()}
    written_at = writer_utc()
    require(stamp(p["recorded_at"]) <= stamp(pre["writer_observed_at"]) <=
            stamp(d["observation"]["started_at"]) <= stamp(d["observation"]["observed_at"]) <=
            stamp(d["writer_observed_at"]) <= stamp(closed["ended_at"]) <=
            stamp(closed["closed_at"]) <= stamp(written_at), "invalid writer/owner phase window")
    return write_new(path, {"schema": SCHEMA, "phase": "sealed", "pre": d["pre"],
                           "during": receipt(Path(during_path).absolute()), "provenance": p,
                           "observation": d["observation"], "closure": closed, "raw": raw,
                           "observed_custody": observed, "writer_observed_at": written_at})
