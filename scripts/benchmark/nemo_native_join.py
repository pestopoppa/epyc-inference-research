"""Pure prospective native NEMO joins. Ambiguous provenance stays UNKNOWN.

No profiler execution, grouping by idle gaps, grading, or historical sidecar reconstruction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "epyc.autokernel.nemo_native_join.v1"
CAPTURE_SCHEMA = "epyc.autokernel.nemo_native_capture.v1"
ROLES = ("run", "native", "markers", "hip", "kernels", "agents")


class JoinError(ValueError):
    """Input bytes or identity cannot support the requested join."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise JoinError(f"{label} is not an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise JoinError(f"{label} is not an integer") from exc
    if str(result) != str(value) or result < minimum:
        raise JoinError(f"{label} is not a canonical integer")
    return result


def _interval(row: Mapping, start: str, end: str) -> tuple[int, int]:
    a, b = _int(row[start], start), _int(row[end], end)
    if b < a:
        raise JoinError("reversed timestamp interval")
    return a, b


def union_ns(intervals: list[tuple[int, int]]) -> int:
    """Device busy time; overlapping dispatches do not add time twice."""
    total = 0
    lo = hi = None
    for a, b in sorted(intervals):
        if b < a:
            raise JoinError("reversed device interval")
        if hi is None:
            lo, hi = a, b
        elif a > hi:
            total += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return total + (0 if hi is None else hi - lo)


def _process(row: Mapping) -> tuple[int, int]:
    return _int(row["pid"], "pid", 1), _int(row["process_start_ticks"], "birth", 1)


def _call_key(row: Mapping) -> tuple:
    return (
        *_process(row),
        _int(row["tid"], "tid", 1),
        row["api"],
        _int(row["call_id"], "call_id", 1),
    )


def join_raw(
    run: Mapping, native: list[dict], markers: list[dict], hip: list[dict], kernels: list[dict]
) -> dict:
    """Join recorded facts only; this does not qualify a capture or emit beliefs."""
    process = (_int(run["pid"], "run.pid", 1), _int(run["pid_birth_ticks"], "run.birth", 1))
    if not isinstance(run.get("host_boot_id"), str) or not run["host_boot_id"]:
        raise JoinError("missing host boot identity")
    marks = run.get("marks")
    if not isinstance(marks, list) or not marks:
        raise JoinError("missing original HTTP request windows")
    requests = {}
    for mark in marks:
        req = mark.get("response_request_id")
        if not isinstance(req, str) or not req or req in requests:
            raise JoinError("missing or repeated response request identity")
        if mark.get("clock") != "clock_gettime(CLOCK_MONOTONIC)":
            raise JoinError("unverified request/profiler clock domain")
        a, b = _interval(mark, "start_monotonic_ns", "end_monotonic_ns")
        requests[req] = {
            "request_id": req,
            "label": mark["label"],
            "start_ns": a,
            "end_ns": b,
            "predicted_n": mark.get("predicted_n"),
            "calls": [],
            "tokens": [],
            "rounds": [],
            "unknown": [],
        }
        if mark.get("err"):
            requests[req]["unknown"].append("HTTP request failed")
    begins, ends = defaultdict(list), defaultdict(list)
    tokens, accepts = [], []
    for event in native:
        if event.get("schema") != "epyc.native_profile.v1" or _process(event) != process:
            raise JoinError("foreign native schema/process/birth")
        _int(event["monotonic_ns"], "native timestamp")
        kind = event.get("event")
        if kind in ("encode_begin", "decode_begin", "encode_end", "decode_end"):
            if event.get("api") not in ("encode", "decode") or not kind.startswith(event["api"]):
                raise JoinError("native API kind differs from event")
            (begins if kind.endswith("begin") else ends)[_call_key(event)].append(event)
        elif kind == "sampled_token":
            tokens.append(event)
        elif kind == "spec_accept":
            accepts.append(event)
        else:
            raise JoinError("unsupported native event")
    calls = []
    for key in sorted(set(begins) | set(ends)):
        starts, stops = begins[key], ends[key]
        call = {
            "key": list(key),
            "unknown": [],
            "request_id": None,
            "dispatches": [],
            "hip_rows": [],
            "emitted": [],
            "busy_ns": None,
            "kernel_sum_ns": None,
        }
        calls.append(call)
        if len(starts) != 1 or len(stops) != 1:
            call["unknown"].append("missing or repeated native begin/end")
            # Never lose a defective call merely because it cannot be paired.
            for event in starts + stops:
                for req in requests.values():
                    if req["start_ns"] <= event["monotonic_ns"] <= req["end_ns"]:
                        req["unknown"].append(call["unknown"][0])
            for start in starts:
                for slot in start.get("slots", []):
                    if slot.get("request_id") in requests:
                        requests[slot["request_id"]]["unknown"].append(call["unknown"][0])
            continue
        start, stop = starts[0], stops[0]
        call.update(begin=start, end=stop)
        if stop.get("rc") != 0 or stop["monotonic_ns"] < start["monotonic_ns"]:
            call["unknown"].append("failed or reversed native API")
        rows = start.get("rows")
        if not isinstance(rows, list) or len(rows) != _int(start.get("n_tokens"), "batch size"):
            call["unknown"].append("native batch rows missing")
        elif [r.get("index") for r in rows] != list(range(len(rows))):
            call["unknown"].append("native batch row indices differ")
        slots = start.get("slots")
        if not isinstance(slots, list) or len(slots) != 1 or start.get("scope_id", 0) <= 0:
            call["unknown"].append("unassociated or multi-slot scope")
        else:
            slot = slots[0]
            req = slot.get("request_id")
            if req not in requests:
                call["unknown"].append("response identity is outside captured requests")
            else:
                call.update(request_id=req, slot=slot)
                requests[req]["calls"].append(len(calls) - 1)
                window = requests[req]
                if not (
                    window["start_ns"]
                    <= start["monotonic_ns"]
                    <= stop["monotonic_ns"]
                    <= window["end_ns"]
                ):
                    call["unknown"].append("native API outside exact HTTP window")
                ctx = start.get("ctx")
                role = (
                    "target"
                    if ctx == slot.get("target_ctx")
                    else ("drafter" if ctx == slot.get("drafter_ctx") else "unknown")
                )
                call["context_role"] = role
                if role == "unknown" or slot.get("server_phase") not in ("prompt", "generation"):
                    call["unknown"].append("context/phase not explicitly associated")
        label = f"epyc.{key[3]}.{key[4]}"
        ranges = [
            r
            for r in markers
            if r.get("Function") == label
            and _int(r.get("Process_Id"), "marker pid", 1) == key[0]
            and _int(r.get("Thread_Id"), "marker tid", 1) == key[2]
        ]
        if len(ranges) != 1:
            call["unknown"].append("missing or repeated exact ROCTX marker")
        else:
            a, b = _interval(ranges[0], "Start_Timestamp", "End_Timestamp")
            call["marker"] = dict(ranges[0])
            if not a <= start["monotonic_ns"] <= stop["monotonic_ns"] <= b:
                call["unknown"].append("ROCTX/native clock containment differs")
    hip_by_correlation = defaultdict(list)
    for index, row in enumerate(hip):
        a, b = _interval(row, "Start_Timestamp", "End_Timestamp")
        pid, tid = _int(row["Process_Id"], "HIP pid", 1), _int(row["Thread_Id"], "HIP tid", 1)
        corr = str(_int(row["Correlation_Id"], "HIP correlation"))
        owners = []
        for ci, call in enumerate(calls):
            marker = call.get("marker")
            if marker and call["key"][:3:2] == [pid, tid]:
                lo, hi = _interval(marker, "Start_Timestamp", "End_Timestamp")
                if lo <= a <= b <= hi:
                    owners.append(ci)
        hip_by_correlation[corr].append((index, owners, pid))
        for ci in owners:
            calls[ci]["hip_rows"].append(index)
            if len(owners) != 1:
                calls[ci]["unknown"].append("HIP API belongs to multiple native markers")
    unbound = []
    seen_dispatches = set()
    for row in kernels:
        a, b = _interval(row, "Start_Timestamp", "End_Timestamp")
        corr = str(_int(row["Correlation_Id"], "dispatch correlation"))
        identity = tuple(
            row.get(k)
            for k in (
                "Agent_Id",
                "Queue_Id",
                "Kernel_Id",
                "Correlation_Id",
                "Start_Timestamp",
                "End_Timestamp",
            )
        )
        candidates = hip_by_correlation.get(corr, [])
        owner = candidates[0][1] if len(candidates) == 1 and candidates[0][2] == process[0] else []
        if len(owner) != 1 or identity in seen_dispatches:
            unbound.append(dict(row))
            for req in requests.values():
                if a < req["end_ns"] and b > req["start_ns"]:
                    req["unknown"].append("unbound/ambiguous/duplicate dispatch intersects request")
        else:
            calls[owner[0]]["dispatches"].append(dict(row))
            if not calls[owner[0]].get("request_id"):
                for req in requests.values():
                    if a < req["end_ns"] and b > req["start_ns"]:
                        req["unknown"].append("unassociated native GPU work intersects request")
        seen_dispatches.add(identity)
    for call in calls:
        dispatches = call["dispatches"]
        if dispatches:
            agents = {r["Agent_Id"] for r in dispatches}
            if len(agents) != 1:
                call["unknown"].append("native API spans multiple GPU agents")
            else:
                spans = [_interval(r, "Start_Timestamp", "End_Timestamp") for r in dispatches]
                call["busy_ns"] = union_ns(spans)
                call["kernel_sum_ns"] = sum(b - a for a, b in spans)
        elif call.get("request_id"):
            call["unknown"].append("no correlated GPU dispatch for native API")
        if call.get("request_id"):
            requests[call["request_id"]]["unknown"].extend(call["unknown"])
    for call in calls:
        if call.get("begin") and not call.get("request_id"):
            for req in requests.values():
                if req["start_ns"] <= call["begin"]["monotonic_ns"] <= req["end_ns"]:
                    req["unknown"].append("unassociated native API inside request")
    for event in tokens:
        req = requests.get(event.get("request_id"))
        if req is None:
            raise JoinError("emitted token has no captured response identity")
        req["tokens"].append(dict(event))
    for req in requests.values():
        emitted = sorted(req["tokens"], key=lambda r: _int(r["ordinal"], "token ordinal", 1))
        if [t["ordinal"] for t in emitted] != list(range(1, len(emitted) + 1)):
            req["unknown"].append("emitted ordinal sequence missing/repeated")
        if req["predicted_n"] != len(emitted):
            req["unknown"].append("native emitted vector does not reconcile response count")
        for token in emitted:
            if not (req["start_ns"] <= token["monotonic_ns"] <= req["end_ns"]):
                req["unknown"].append("emitted token outside request")
            _int(token["token"], "emitted token")
            candidates = [
                calls[ci]
                for ci in req["calls"]
                if calls[ci].get("context_role") == "target"
                and calls[ci]["key"][3] == "decode"
                and calls[ci]["key"][2] == token.get("tid")
                and calls[ci]["slot"].get("slot") == token.get("slot")
                and calls[ci]["slot"].get("task") == token.get("task")
                and calls[ci]["begin"].get("ctx") == token.get("target_ctx")
                and calls[ci]["slot"].get("generated_before") == token["ordinal"] - 1
                and calls[ci]["end"]["monotonic_ns"] <= token["monotonic_ns"] <= req["end_ns"]
            ]
            if len(candidates) == 1:
                candidate = candidates[0]
                candidate["emitted"].append(token)
                if (
                    candidate["slot"].get("server_phase") == "generation"
                    and candidate["begin"]["n_tokens"] == 1
                ):
                    rows = candidate["begin"]["rows"]
                    previous = [t for t in emitted if t["ordinal"] == token["ordinal"] - 1]
                    if (
                        len(previous) != 1
                        or rows[0].get("token") != previous[0]["token"]
                        or rows[0].get("pos") is None
                        or rows[0].get("seq_ids") != [candidate["slot"]["slot"]]
                    ):
                        req["unknown"].append(
                            "plain native input does not bind previous emitted token/position/sequence"
                        )
            else:
                # A later speculative group may bind this token, but never by counts alone.
                matching = [
                    e
                    for e in accepts
                    if e.get("request_id") == req["request_id"]
                    and e.get("slot") == token.get("slot")
                    and e.get("task") == token.get("task")
                    and e.get("target_ctx") == token.get("target_ctx")
                    and e.get("tid") == token.get("tid")
                    and e.get("generated_before", -1)
                    < token["ordinal"]
                    <= e.get("generated_before", -1)
                    + len(e.get("accepted_tokens_including_bonus", []))
                ]
                if len(matching) != 1:
                    req["unknown"].append("emitted token native call/group is ambiguous")
        accepted = [e for e in accepts if e.get("request_id") == req["request_id"]]
        if len({e.get("generated_before") for e in accepted}) != len(accepted):
            req["unknown"].append("repeated speculative round identity")
        for event in accepted:
            before = _int(event["generated_before"], "speculative generated_before")
            vector = event.get("accepted_tokens_including_bonus")
            if not isinstance(vector, list) or not vector:
                req["unknown"].append("missing accepted token vector")
                continue
            prefix = [t for t in emitted if before < t["ordinal"] <= before + len(vector)]
            if not prefix or [t["token"] for t in prefix] != vector[: len(prefix)]:
                req["unknown"].append("speculative accepted/emitted token vectors differ")
            group = [
                ci
                for ci in req["calls"]
                if calls[ci]["slot"].get("generated_before") == before
                and calls[ci]["slot"].get("server_phase") == "generation"
            ]
            if not group or not any(calls[ci].get("context_role") == "target" for ci in group):
                req["unknown"].append("speculative vector has no unique native verification group")
            if any(
                calls[ci]["key"][2] != event.get("tid")
                or calls[ci]["slot"].get("slot") != event.get("slot")
                or calls[ci]["slot"].get("task") != event.get("task")
                or calls[ci]["slot"].get("target_ctx") != event.get("target_ctx")
                for ci in group
            ):
                req["unknown"].append("speculative native group identity differs")
            req["rounds"].append(
                {
                    "event": event,
                    "call_indices": group,
                    "emitted": prefix,
                    "per_token_busy_ns": None,
                    "reason": "joint speculative batch; no per-token allocation",
                }
            )
        ordered_calls = sorted(
            (calls[ci] for ci in req["calls"]), key=lambda c: c["begin"]["monotonic_ns"]
        )
        for call in ordered_calls:
            if call["key"][3] != "encode":
                continue
            same_scope = [
                c
                for c in ordered_calls
                if c["key"][2] == call["key"][2]
                and c["begin"].get("scope_id") == call["begin"].get("scope_id")
                and c["begin"].get("ctx") == call["begin"].get("ctx")
                and c["begin"]["monotonic_ns"] > call["begin"]["monotonic_ns"]
            ]
            inject = same_scope[0] if same_scope else None
            if (
                call.get("context_role") != "drafter"
                or not inject
                or inject["key"][3] != "decode"
                or call["unknown"]
                or inject["unknown"]
                or call["begin"]["n_tokens"] != inject["begin"]["n_tokens"]
                or not all(
                    r.get("pos") is not None and r.get("seq_ids") == [call["slot"]["slot"]]
                    for r in inject["begin"]["rows"]
                )
            ):
                req["unknown"].append("encode has no unique successful same-scope injection")
            else:
                call["injection_call_key"] = inject["key"]
        req["unknown"] = sorted(set(req["unknown"]))
        req["status"] = "UNKNOWN" if req["unknown"] else "complete"
    return {
        "schema": SCHEMA,
        "host_boot_id": run["host_boot_id"],
        "pid": process[0],
        "birth": process[1],
        "status": "UNKNOWN" if any(r["unknown"] for r in requests.values()) else "complete",
        "calls": calls,
        "requests": list(requests.values()),
        "unbound_dispatches": unbound,
        "source_run_hash": canonical_hash(run),
        "qualification": "join only; formal capture admission/finalization separate",
    }


def read_capture(manifest: Mapping, root: Path) -> tuple[dict, dict]:
    """Reopen exactly hash-bound raw bytes; never load code or follow an outside path."""
    if manifest.get("schema") != CAPTURE_SCHEMA:
        raise JoinError("unsupported prospective capture manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(ROLES):
        raise JoinError("capture needs exact raw readset roles")
    values, hashes = {}, {}
    root = root.resolve()
    for role in ROLES:
        item = files[role]
        path = root / item["path"]
        if (
            Path(item["path"]).is_absolute()
            or not path.resolve().is_relative_to(root)
            or path.is_symlink()
        ):
            raise JoinError("raw readset path escapes capture root")

        def stable(st):
            return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

        before = stable(path.stat())
        raw = path.read_bytes()
        if stable(path.stat()) != before or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise JoinError("raw readset bytes changed or hash differs")
        hashes[role] = item["sha256"]
        if role == "run":
            values[role] = json.loads(raw)
        elif role == "native":
            values[role] = [json.loads(line) for line in raw.splitlines() if line]
        else:
            values[role] = list(csv.DictReader(raw.decode().splitlines()))
    report = join_raw(*(values[k] for k in ("run", "native", "markers", "hip", "kernels")))
    report["raw_readset"] = hashes
    return report, values


def measurement_values(report: Mapping) -> list[dict]:
    """Only the original three lower-better, single-request attribution metrics.

    There is no keep/ship edge, confidence upgrade, or invented independent-token n.
    Joint speculative work is measured per round, never divided into token identities.
    """
    if report.get("status") != "complete":
        raise JoinError("ambiguous/partial native report cannot emit measurements")
    calls, result = report["calls"], []
    for req in report["requests"]:
        label = req["label"]
        if label in ("P", "M"):
            selected = [
                calls[i]
                for i in req["calls"]
                if calls[i].get("context_role") == "target"
                and calls[i]["slot"]["server_phase"] == "generation"
                and calls[i]["key"][3] == "decode"
            ]
            minimum = 20 if label == "P" else 30
            if len(selected) < minimum or req["rounds"]:
                raise JoinError("original plain-token/decode-step sample requirement not met")
            for call in selected:
                rows = call["begin"]["rows"]
                if (
                    call["begin"]["n_tokens"] != 1
                    or len(call["emitted"]) != 1
                    or call["busy_ns"] is None
                    or rows[0].get("pos") is None
                    or rows[0].get("seq_ids") != [call["slot"]["slot"]]
                ):
                    raise JoinError(
                        "plain token requires a unique explicit native input/output binding"
                    )
            values = [call["busy_ns"] / 1e6 for call in selected]
            metric = "plain_token_busy_ms" if label == "P" else "bolt_decode_step_busy_ms"
        elif label == "S":
            groups = req["rounds"]
            if len(groups) < 10:
                raise JoinError("original speculative-round sample requirement not met")
            seen = set()
            values = []
            for group in groups:
                ids = group["call_indices"]
                if not ids or seen.intersection(ids) or not group["emitted"]:
                    raise JoinError(
                        "speculative round native work is missing or shared ambiguously"
                    )
                seen.update(ids)
                spans = [
                    _interval(d, "Start_Timestamp", "End_Timestamp")
                    for i in ids
                    for d in calls[i]["dispatches"]
                ]
                if not spans:
                    raise JoinError("speculative round has no correlated device work")
                values.append(union_ns(spans) / 1e6)
            metric = "dflash_round_busy_ms"
        else:
            continue  # Warm-up and canary labels are never scored profile rows.
        result.append(
            {
                "metric": metric,
                "label": label,
                "value_ms": sum(values) / len(values),
                "raw_values_ms": values,
                "observed_dependent_units": len(values),
                "request_id": req["request_id"],
                "reps": 1,
                "reps_basis": "one traced request; dependent native calls/rounds, no repeats or noise floor",
            }
        )
    if not result:
        raise JoinError("no original scored L/S,P or B/M requests")
    return result


NATIVE_AUTHORITY = {
    "protocol": "914fd9b116a7d9c84fb10842165069db428d7ab26992c2e281bb9be89cd692ca",
    "original_l": "e746d105ccfa01e7542d918a9aaf6ed3073b0171775f341979895b988a278a4d",
    "libggml": "b89dcc64a3687f0c8e918200046d302d2a1c8cd38e44ece7e86c04db176ee33e",
    "libggml_base": "7b5376ef2a3c9cd4c9aa74b4abb92ca2d979e123999601db6ed98dd639c0c155",
    "libggml_cpu": "74be93aa3e1443217c1e1e18dfff930a8ebbcba13b53455cb4a3f0c25e1b0d8b",
    "profiler_sdk": "44d8548b9e31c7ab4ecad3023878d9b9d8bcf62b69350ba3837c01270d45639c",
    "source_manifest": "e7859f018098bedb9c8951c76a8264ec350f80764f0b4b3fa36022eee9d6b986",
    "server": "83805b23f09b35abb04b53aefdd15fb3e71f5eb55861c2d3945b28a846753e65",
    "libllama": "7d317ac83a4fede24714e044295156e94ed86ffd70ef154acbf88d700bac9453",
    "libggml_hip": "1f3c92e30db1f9c4c675a1627698f46aa3834a8e2e2cf12b7deabfc292070357",
    "profiler": "c753449eb635ecb4d8be794e8b66439b200b252c157555920d260df5cbac767a",
    "roctx": "93db9899dfccfcdac33f5721f81ed1df39251c2b98f795bac36ca563d3e8e7d8",
}
CODE_PROMPT = "Write a Python function that parses a CSV file into a list of dictionaries, with type inference for integer and float columns, and add a docstring and two usage examples."
PROSE_PROMPT = "Explain in a short paragraph why the sky is blue."
BOLT_MODEL = "/mnt/raid0/llm/models/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf"


def artifact_identity(path):
    """Stream bytes with a before/after identity check, never trust supplied hashes."""
    p = Path(path)
    before = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    after = p.stat()

    def key(s):
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    if key(before) != key(after) or not p.is_absolute() or not p.is_file():
        raise JoinError("artifact identity changed while reading")
    return {"path": str(p), "size": after.st_size, "sha256": digest.hexdigest()}


def phase_payloads(pass_name):
    def chat(prompt, count, extra=None):
        return {
            "endpoint": "/v1/chat/completions",
            "payload": {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": count,
                "temperature": 0.0,
                "top_k": 1,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
                **(extra or {}),
            },
        }

    if pass_name == "L":
        return {
            "warm": chat(CODE_PROMPT, 160),
            "S": chat(CODE_PROMPT, 160),
            "P": chat(CODE_PROMPT, 160, {"speculative.n_max": 0}),
        }
    if pass_name == "B":
        return {"warm": chat(PROSE_PROMPT, 48), "M": chat(CODE_PROMPT, 96)}
    raise JoinError("unknown original pass")


def validate_capture_contract(capture, run):
    """Prospective artifact/workload/owner evidence; independent of grade policy."""
    from datetime import datetime

    def require(ok, why):
        if not ok:
            raise JoinError("capture contract: " + why)

    def instant(value):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(parsed.tzinfo is not None, "timezone missing")
        return parsed

    contract = capture.get("capture_contract", {})
    require(
        contract.get("schema") == "epyc.autokernel.nemo_native_contract.v1",
        "missing identity contract",
    )
    require(
        contract.get("purpose") == capture.get("purpose") == "prospective_scored_profile",
        "canary or relabelled purpose",
    )
    digest = canonical_hash(contract)
    require(run.get("capture_contract_sha256") == digest, "source-time contract hash differs")
    require(
        contract.get("native_commit") == "b1e820a0ad8bc2192fc38c3c1beda7abf3a28ecf"
        and contract.get("native_base") == "ffc1bac82eeca6f9099e1ccd9ba49703c460a115",
        "native source era differs",
    )
    require(
        contract.get("analysis_sources") == capture.get("analysis_sources"),
        "analysis not frozen at source time",
    )
    require(instant(contract["declared_at"]) <= instant(run["started_at"]), "late contract")
    analysis_artifacts = contract.get("analysis_artifacts", {})
    require(
        set(analysis_artifacts) == set(contract["analysis_sources"]),
        "missing analysis artifact map",
    )
    for name, item in analysis_artifacts.items():
        require(
            artifact_identity(item["path"]) == item
            and item["sha256"] == contract["analysis_sources"][name],
            "analysis module bytes differ",
        )
    artifacts = contract.get("artifacts", {})
    required = set(NATIVE_AUTHORITY) | {
        "producer",
        "claim_module",
        "claim_schemas",
        "worktree_module",
        "contract_module",
        "interpreter",
        "launcher",
        "capture_source_manifest",
        "module_closure",
        "model",
        "template",
    }
    if contract.get("pass") == "L":
        required.add("draft_model")
    require(set(artifacts) == required, "missing or extra artifact identities")
    require(run.get("source_time_artifacts") == artifacts, "source-time artifact map differs")
    for role, item in artifacts.items():
        require(artifact_identity(item["path"]) == item, "artifact bytes differ: " + role)
        if role in NATIVE_AUTHORITY:
            require(
                item["sha256"] == NATIVE_AUTHORITY[role],
                "unapproved source/build/protocol: " + role,
            )
    closure = json.loads(Path(artifacts["module_closure"]["path"]).read_bytes())
    require(isinstance(closure, dict) and len(closure) >= 12, "missing imported module closure")
    for path, expected_sha in closure.items():
        require(artifact_identity(path)["sha256"] == expected_sha, "imported module bytes differ")
    for role in ("claim_module", "claim_schemas", "worktree_module"):
        require(
            closure.get(artifacts[role]["path"]) == artifacts[role]["sha256"],
            "module outside pinned closure",
        )
    server = contract["server_argv"]
    require(
        isinstance(server, list) and server[0] == artifacts["server"]["path"],
        "server identity differs",
    )
    port = server[server.index("--port") + 1]
    if contract["pass"] == "L":
        original = json.loads(Path(artifacts["original_l"]["path"]).read_bytes())["argv"]
        index = next(i for i, x in enumerate(original) if x.endswith("/llama-server"))
        expected = [server[0]] + original[index + 1 :]
        for flag, value in (("--port", port), ("-np", "1"), ("-c", "16384")):
            expected[expected.index(flag) + 1] = value
        if "--slot-save-path" in expected:
            index = expected.index("--slot-save-path")
            del expected[index : index + 2]
        require(expected == server, "original L argv differs")
        for role, flag in (
            ("model", "-m"),
            ("draft_model", "-md"),
            ("template", "--chat-template-file"),
        ):
            require(
                server[server.index(flag) + 1] == artifacts[role]["path"],
                "L model/template binding differs",
            )
    else:
        expected = [
            server[0],
            "-m",
            BOLT_MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--batch-size",
            "2048",
            "--ubatch-size",
            "512",
            "--ctx-size",
            "16384",
            "--parallel",
            "1",
            "-ngl",
            "999",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--flash-attn",
            "on",
            "--jinja",
            "-t",
            "8",
        ]
        require(
            expected == server and artifacts["model"]["path"] == BOLT_MODEL,
            "original B model/argv differs",
        )
        require(
            artifacts["template"] == artifacts["model"]
            and contract.get("template_mode") == "embedded_gguf",
            "B embedded template not bound to actual model",
        )
    require(
        all(m.get("runtime_server_argv") == server for m in run["marks"]),
        "actual running server argv differs",
    )
    require(
        run.get("server_argv") == server
        and run.get("argv") == contract.get("argv")
        and run.get("env") == contract.get("env"),
        "actual argv/env differs",
    )
    require(contract["argv"][-len(server) :] == server, "wrapper does not launch bound server")
    for flag in (
        "--kernel-trace",
        "--memory-copy-trace",
        "--hip-trace",
        "--marker-trace",
        "--stats",
    ):
        require(flag in contract["argv"][: -len(server)], "capture API missing " + flag)
    require(
        contract["env"]["set"].get("LLAMA_NATIVE_ROCTX_LIBRARY") == artifacts["roctx"]["path"]
        and "GGML_CUDA_DISABLE_GRAPHS" in contract["env"]["unset"]
        and "GGML_CUDA_DISABLE_GRAPHS" not in contract["env"]["set"],
        "ROCTX/graph env differs",
    )
    for mark in run["marks"]:
        expected_runtime = {
            role: artifact_identity(str(Path(artifacts[role]["path"]).resolve()))
            for role in (
                "server",
                "libllama",
                "libggml_hip",
                "libggml",
                "libggml_base",
                "libggml_cpu",
                "roctx",
                "profiler_sdk",
            )
        }
        require(
            mark.get("runtime_artifacts") == expected_runtime,
            "actual loaded image/DSO identities differ",
        )
    require(
        artifacts["contract_module"]["sha256"]
        == contract["analysis_sources"]["nemo_native_join.py"],
        "producer contract module differs from frozen analysis",
    )
    wrapper = contract["argv"][: -len(server)]
    require(wrapper and wrapper[0] == "env", "unexpected launch wrapper")
    index, actual_unset, actual_sets = 1, [], {}
    while index < len(wrapper):
        if wrapper[index] == "-u":
            actual_unset.append(wrapper[index + 1])
            index += 2
        elif "=" in wrapper[index] and not wrapper[index].startswith("--"):
            key, value = wrapper[index].split("=", 1)
            require(key not in actual_sets, "duplicate environment binding")
            actual_sets[key] = value
            index += 1
        else:
            break
    require(
        actual_unset == contract["env"]["unset"] and actual_sets == contract["env"]["set"],
        "environment launch prefix differs",
    )
    cpus = contract.get("cpu_list")
    require(cpus == "184-191", "original GPU CPU placement differs")
    expected_wrapper = [
        "numactl",
        "--membind=3",
        "--",
        "taskset",
        "-c",
        cpus,
        artifacts["interpreter"]["path"],
        artifacts["profiler"]["path"],
        "--kernel-trace",
        "--memory-copy-trace",
        "--hip-trace",
        "--marker-trace",
        "--stats",
        "--output-format",
        "csv",
        "-d",
        contract["trace_dir"],
        "-o",
        "prof",
        "--",
    ]
    require(wrapper[index:] == expected_wrapper, "profiler/placement wrapper differs")
    expected_phases = phase_payloads(contract["pass"])
    require(contract.get("phases") == expected_phases, "original workload/payload differs")
    require(
        [m["label"] for m in run["marks"]] == list(expected_phases), "phase order/coverage differs"
    )
    opened, released = capture["device_claim_open"], capture["device_claim_released"]
    for field in (
        "schema",
        "claim_id",
        "device_id",
        "lock_path",
        "holder_pid",
        "holder_start_ticks",
        "holder_boot_id",
        "campaign_id",
        "purpose",
        "acquired_at",
        "expires_at",
    ):
        require(
            opened.get(field) is not None and opened.get(field) == released.get(field),
            "claim identity differs: " + field,
        )
    require(
        all(
            type(opened[key]) is int and opened[key] > 0
            for key in ("holder_pid", "holder_start_ticks")
        ),
        "invalid claim PID/birth",
    )
    require(
        opened["lock_path"] == "/mnt/raid0/llm/tmp/gpu_device.mi210_0.lock",
        "foreign device lock identity",
    )
    require(
        opened["schema"] == "epyc.autokernel.device_claim_receipt.v1"
        and opened.get("state") == released.get("state") == "held",
        "claim was not held",
    )
    require(
        opened["device_id"]
        == capture["agent_binding"]["device_id"]
        == contract.get("device_id")
        == "mi210_0",
        "physical device differs",
    )
    require(
        opened["campaign_id"] == capture["campaign_id"] == contract["campaign_id"],
        "claim campaign differs",
    )
    require(
        run.get("capture_owner")
        == {
            key: opened[key]
            for key in (
                "holder_pid",
                "holder_start_ticks",
                "holder_boot_id",
                "claim_id",
                "device_id",
            )
        },
        "source-time owner differs",
    )
    require(opened["holder_boot_id"] == run["host_boot_id"], "claim boot differs")
    acquired, expires, release = map(
        instant, (opened["acquired_at"], opened["expires_at"], released["released_at"])
    )
    start, end = map(instant, (run["started_at"], run["finished_at"]))
    require(
        acquired <= start < end <= release <= expires and instant(run["drained_at"]) <= release,
        "run/drain outside held claim window",
    )
    for mark in run["marks"]:
        phase = expected_phases[mark["label"]]
        require(
            {"endpoint": mark.get("endpoint"), "payload": mark.get("payload")} == phase
            and mark.get("capture_contract_sha256") == digest,
            "actual HTTP/template payload differs",
        )
        require(
            start <= instant(mark["started_at"]) < instant(mark["ended_at"]) <= end,
            "request outside captured/claimed window",
        )


def qualify_capture(capture: Mapping, values: Mapping) -> None:
    """Validate prospective custody, not a new measurement grading rule."""
    from datetime import datetime

    run = values["run"]
    validate_capture_contract(capture, run)
    if capture.get("purpose") != "prospective_scored_profile":
        raise JoinError("canary/legacy capture is not a scored prospective profile")
    if (
        capture.get("status") != "complete"
        or run.get("status") != "capture-complete-unvalidated"
        or run.get("trip")
        or not run.get("teardown", {}).get("verified_dead")
        or run["teardown"].get("returncode") != 0
        or run.get("server_core_limit") != [0, 0]
    ):
        raise JoinError("partial/failed/unflushed native capture")
    if run.get("cpu_only_instrumentation_smoke") or run.get("tiny_gpu_instrumentation_smoke"):
        raise JoinError("R11 instrumentation canary cannot become original-profile evidence")
    try:
        declared = datetime.fromisoformat(capture["declared_at"].replace("Z", "+00:00"))
        started = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
        if declared.tzinfo is None or started.tzinfo is None or declared > started:
            raise JoinError("analysis declaration did not precede capture")
    except (KeyError, ValueError) as exc:
        raise JoinError("missing prospective declaration time") from exc
    if (
        capture.get("started_at") != run["started_at"]
        or capture.get("ended_at") != run["finished_at"]
    ):
        raise JoinError("capture/run timestamp identities differ")
    binding = capture.get("agent_binding", {})
    required = {"device_id", "agent_id", "drm_render_minor", "location_id", "gpu_id"}
    if not isinstance(binding, Mapping) or set(binding) != required:
        raise JoinError("missing explicit physical device/agent binding")
    agents = [
        r
        for r in values["agents"]
        if r.get("Agent_Type") == "GPU"
        and _int(r.get("Node_Id"), "agent node") == binding["agent_id"]
    ]
    if len(agents) != 1 or any(
        _int(agents[0].get(column), column) != binding[key]
        for key, column in (
            ("drm_render_minor", "Drm_Render_Minor"),
            ("location_id", "Location_Id"),
            ("gpu_id", "Gpu_Id"),
        )
    ):
        raise JoinError("physical GPU binding does not rederive from original agent bytes")
    if any(_int(r["Agent_Id"], "dispatch agent") != binding["agent_id"] for r in values["kernels"]):
        raise JoinError("mixed/foreign dispatch device")
    samples = run.get("during_request_gpu_proofs", [])
    for mark in run["marks"]:
        if mark["label"] not in ("S", "P", "M"):
            continue
        valid = [
            s
            for s in samples
            if s.get("pid") == run["pid"]
            and s.get("kfd_present") is True
            and s.get("vram_bytes", 0) > 0
            and s.get("hip_library_maps")
            and mark["start_monotonic_ns"]
            <= s["sample_begin_monotonic_ns"]
            <= s["sample_end_monotonic_ns"]
            <= mark["end_monotonic_ns"]
        ]
        if len(valid) < 5:
            raise JoinError("original minimum five during-request residency observations missing")
