#!/usr/bin/env python3
"""Convert benchmark/log/report JSONL into a bounded closed-loop observation surface.

The surface is intentionally small and deterministic:

- No network access
- No model inference
- No runtime timestamps
- Stable IDs derived from content + source position
- OTEL-like envelopes with a single parent chain per loop

Supported input kinds:

- benchmark: benchmark result records
- log: log/event records
- report: report/summary records

The module exposes pure conversion/analyzer helpers and a CLI with two
subcommands:

- ``convert`` writes normalized observation JSONL
- ``analyze`` summarizes a previously converted observation stream
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import blake2s
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_NAME = "epyc.observation.v1"
OBSERVATION_PREFIX = "epyc.halo.closed_loop"

_BENCHMARK_HINTS = {
    "question_id",
    "suite",
    "outcome",
    "score",
    "prompt",
    "response",
    "rewards",
    "tokens_per_second",
    "completion_tokens",
}
_LOG_HINTS = {
    "level",
    "message",
    "logger",
    "event",
    "severity",
    "thread",
}
_REPORT_HINTS = {
    "report_id",
    "summary",
    "verdict",
    "recommendation",
    "title",
    "sections",
    "decision",
}

_BENCHMARK_ATTRS = (
    "run_id",
    "question_id",
    "suite",
    "tier",
    "role",
    "model",
    "config",
    "outcome",
    "score",
    "f1",
    "precision",
    "recall",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_second",
    "total_time_ms",
    "elapsed_seconds",
    "label_4class",
    "source",
)
_LOG_ATTRS = (
    "run_id",
    "job_id",
    "session_id",
    "level",
    "severity",
    "logger",
    "message",
    "event",
    "component",
    "step",
    "source",
)
_REPORT_ATTRS = (
    "run_id",
    "report_id",
    "title",
    "status",
    "verdict",
    "decision",
    "summary",
    "recommendation",
    "source",
)


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    line_no: int
    payload: dict[str, Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: str, *, size: int = 16) -> str:
    return blake2s(value.encode("utf-8"), digest_size=size).hexdigest()


def _stable_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_stable_scalar(item) for item in value]
    if isinstance(value, tuple):
        return [_stable_scalar(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable_scalar(val) for key, val in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return str(value)


def _iter_input_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".jsonl")
        else:
            yield path


def read_jsonl(path: Path) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(SourceRecord(path=path, line_no=line_no, payload=payload))
    return records


def load_sources(paths: Iterable[Path]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for path in _iter_input_files(paths):
        records.extend(read_jsonl(path))
    records.sort(key=lambda record: (str(record.path), record.line_no))
    return records


def detect_record_kind(payload: dict[str, Any]) -> str:
    explicit = payload.get("record_type")
    if isinstance(explicit, str) and explicit in {"benchmark", "log", "report"}:
        return explicit

    keys = set(payload)
    benchmark_hits = len(keys & _BENCHMARK_HINTS)
    log_hits = len(keys & _LOG_HINTS)
    report_hits = len(keys & _REPORT_HINTS)

    ranked = sorted(
        [("benchmark", benchmark_hits), ("log", log_hits), ("report", report_hits)],
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    top_kind, top_score = ranked[0]
    if top_score == 0:
        raise ValueError("cannot infer record kind")
    if sum(score == top_score for _, score in ranked) > 1:
        raise ValueError(f"ambiguous record kind: {sorted(kind for kind, score in ranked if score == top_score)}")
    return top_kind


def _subject_fields(kind: str, payload: dict[str, Any]) -> list[str]:
    for key in ("run_id", "batch_id", "job_id", "session_id", "report_id", "question_id", "trace_id"):
        if key in payload and payload[key] not in (None, ""):
            return [f"{key}={_stable_json(_stable_scalar(payload[key]))}"]
    return [f"kind={kind}"]


def loop_key(kind: str, payload: dict[str, Any]) -> str:
    return "|".join(_subject_fields(kind, payload))


def _select_attributes(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in keys:
        if key in payload and payload[key] not in (None, ""):
            attrs[key] = _stable_scalar(payload[key])
    return attrs


def _benchmark_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    attrs = _select_attributes(payload, _BENCHMARK_ATTRS)
    if "prompt" in payload and isinstance(payload["prompt"], str):
        attrs["prompt_preview"] = payload["prompt"][:160]
    if "response" in payload and isinstance(payload["response"], str):
        attrs["response_preview"] = payload["response"][:160]
    return attrs


def _log_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    attrs = _select_attributes(payload, _LOG_ATTRS)
    if "message" in payload and isinstance(payload["message"], str):
        attrs["message"] = payload["message"][:240]
    return attrs


def _report_attributes(payload: dict[str, Any]) -> dict[str, Any]:
    attrs = _select_attributes(payload, _REPORT_ATTRS)
    if "summary" in payload and isinstance(payload["summary"], str):
        attrs["summary"] = payload["summary"][:240]
    if "recommendation" in payload and isinstance(payload["recommendation"], str):
        attrs["recommendation"] = payload["recommendation"][:240]
    return attrs


def build_observation(record: SourceRecord) -> dict[str, Any]:
    kind = detect_record_kind(record.payload)
    attributes_map = {
        "benchmark": _benchmark_attributes,
        "log": _log_attributes,
        "report": _report_attributes,
    }
    attributes = attributes_map[kind](record.payload)
    loop = loop_key(kind, record.payload)
    canonical_payload = _stable_json(record.payload)
    payload_hash = _digest(canonical_payload, size=8)
    observation_id = _digest(
        "|".join(
            [
                SCHEMA_NAME,
                kind,
                loop,
                f"{record.path}:{record.line_no}",
                payload_hash,
            ]
        )
    )
    trace_id = _digest(f"{loop}|trace", size=16)
    span_id = _digest(f"{observation_id}|span", size=8)
    return {
        "schema": SCHEMA_NAME,
        "kind": kind,
        "name": f"{OBSERVATION_PREFIX}.{kind}",
        "observation_id": observation_id,
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": None,
        "sequence": None,
        "loop_id": _digest(loop, size=8),
        "loop_key": loop,
        "source": {"path": str(record.path), "line": record.line_no},
        "attributes": attributes,
        "payload_hash": payload_hash,
    }


def convert_records(records: Iterable[SourceRecord]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    previous_span_by_loop: dict[str, str] = {}
    sequence_by_loop: dict[str, int] = defaultdict(int)
    for record in records:
        observation = build_observation(record)
        loop = observation["loop_key"]
        observation["sequence"] = sequence_by_loop[loop]
        observation["parent_span_id"] = previous_span_by_loop.get(loop)
        previous_span_by_loop[loop] = observation["span_id"]
        sequence_by_loop[loop] += 1
        converted.append(observation)
    return converted


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_stable_json(row))
            handle.write("\n")


def read_observations(paths: Iterable[Path]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for path in _iter_input_files(paths):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_no}: expected JSON object")
                observations.append(payload)
    observations.sort(key=lambda row: (row.get("loop_key", ""), row.get("sequence", 0), row.get("observation_id", "")))
    return observations


def analyze_observations(observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    kind_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    benchmark_outcomes: Counter[str] = Counter()
    log_levels: Counter[str] = Counter()
    report_verdicts: Counter[str] = Counter()
    loops: dict[str, set[str]] = defaultdict(set)
    loop_sizes: Counter[str] = Counter()

    for observation in observations:
        total += 1
        kind = str(observation.get("kind", "unknown"))
        loop_key_value = str(observation.get("loop_key", ""))
        kind_counts[kind] += 1
        loop_sizes[loop_key_value] += 1
        source = observation.get("source", {})
        if isinstance(source, dict):
            source_counts[str(source.get("path", ""))] += 1

        attrs = observation.get("attributes", {})
        if not isinstance(attrs, dict):
            continue
        loops[loop_key_value].add(kind)
        if kind == "benchmark":
            benchmark_outcomes[str(attrs.get("outcome", "unknown"))] += 1
        elif kind == "log":
            log_levels[str(attrs.get("level", attrs.get("severity", "unknown")))] += 1
        elif kind == "report":
            report_verdicts[str(attrs.get("verdict", attrs.get("status", "unknown")))] += 1

    closed_loops = sorted(loop for loop, kinds in loops.items() if {"benchmark", "log", "report"}.issubset(kinds))
    loop_cardinality = {
        "total": len(loop_sizes),
        "closed": len(closed_loops),
        "max_observations": max(loop_sizes.values(), default=0),
        "min_observations": min(loop_sizes.values(), default=0),
    }
    return {
        "schema": SCHEMA_NAME,
        "total_observations": total,
        "kind_counts": dict(sorted(kind_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "benchmark_outcomes": dict(sorted(benchmark_outcomes.items())),
        "log_levels": dict(sorted(log_levels.items())),
        "report_verdicts": dict(sorted(report_verdicts.items())),
        "loop_cardinality": loop_cardinality,
        "closed_loops": closed_loops,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bounded closed-loop observation surface")
    sub = parser.add_subparsers(dest="command", required=True)

    convert = sub.add_parser("convert", help="Convert benchmark/log/report JSONL into observations")
    convert.add_argument("inputs", nargs="+", type=Path, help="Input JSONL files or directories")
    convert.add_argument("--output", "-o", required=True, type=Path, help="Observation JSONL output")

    analyze = sub.add_parser("analyze", help="Summarize observation JSONL")
    analyze.add_argument("inputs", nargs="+", type=Path, help="Observation JSONL files or directories")
    analyze.add_argument("--output", "-o", type=Path, help="Write analysis JSON here instead of stdout")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "convert":
        source_records = load_sources(args.inputs)
        observations = convert_records(source_records)
        write_jsonl(args.output, observations)
        return 0

    observations = read_observations(args.inputs)
    summary = analyze_observations(observations)
    rendered = _stable_json(summary)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
