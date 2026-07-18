"""No-inference comparison materializer for existing ODL/PaddleOCR artifacts.

This module reads artifacts that already exist on disk:

* ``*_row_set.json`` files emitted by ``adapter run`` / ``adapter run-model``.
* raw OmniDocBench ``*_metric_result.json`` files, with an explicit engine name.
* prediction directories whose metric result can be inferred from the current
  ``<prediction_dir_basename>_quick_match_metric_result.json`` convention.
* postprocess summary JSONs that carry ``original_metric_rows`` and
  ``postprocessed_metric_rows``.

It never generates predictions, runs the bench scorer, or launches inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import bootstrap
from .adapter import OdlBenchAdapter
from .schemas import (
    METRIC_READING_ORDER,
    METRIC_SPEED,
    METRIC_STRUCTURAL,
    METRIC_TABLE,
    MetricRow,
)

SCHEMA = "odl_bench.existing_comparison.v1"
JSON_NAME = "existing_comparison.json"
MARKDOWN_NAME = "existing_comparison.md"

METRIC_ORDER = (METRIC_STRUCTURAL, METRIC_TABLE, METRIC_READING_ORDER, METRIC_SPEED)
METRIC_LABELS = {
    METRIC_STRUCTURAL: "structural Edit_dist (lower)",
    METRIC_TABLE: "table TEDS (higher)",
    METRIC_READING_ORDER: "reading-order Edit_dist (lower)",
    METRIC_SPEED: "latency_ms median (lower)",
}
METRIC_DIRECTIONS = {
    METRIC_STRUCTURAL: "lower_better",
    METRIC_TABLE: "higher_better",
    METRIC_READING_ORDER: "lower_better",
    METRIC_SPEED: "lower_better",
}


def parse_engine_path_spec(spec: str) -> tuple[str | None, Path]:
    """Parse ``ENGINE=PATH`` or bare ``PATH`` CLI specs."""
    if "=" not in spec:
        return None, Path(spec)
    engine, path = spec.split("=", 1)
    engine = engine.strip()
    path = path.strip()
    if not engine or not path:
        raise ValueError(f"invalid ENGINE=PATH spec: {spec!r}")
    return engine, Path(path)


def parse_engine_path_specs(specs: list[str] | tuple[str, ...]) -> list[tuple[str | None, Path]]:
    return [parse_engine_path_spec(spec) for spec in specs]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _metric_row_from_dict(
    row: dict[str, Any],
    *,
    default_engine: str | None = None,
    engine_override: str | None = None,
) -> MetricRow:
    engine = str(engine_override or row.get("engine") or default_engine or "")
    if not engine:
        raise ValueError(f"metric row is missing engine: {row!r}")
    value = row.get("value")
    return MetricRow(
        engine=engine,
        metric_family=str(row.get("metric_family") or ""),
        metric_name=str(row.get("metric_name") or ""),
        value=float(value) if isinstance(value, (int, float)) else None,
        n=int(row.get("n") or 0),
        detail=str(row.get("detail") or ""),
    )


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _rows_from_row_dicts(
    rows: Any,
    *,
    default_engine: str | None = None,
    engine_override: str | None = None,
) -> list[MetricRow]:
    if not isinstance(rows, list):
        return []
    return [
        _metric_row_from_dict(row, default_engine=default_engine, engine_override=engine_override)
        for row in rows
        if isinstance(row, dict)
    ]


def rows_from_artifact(
    path: str | Path,
    *,
    engine: str | None = None,
) -> tuple[list[MetricRow], dict[str, Any]]:
    """Read one existing artifact and return metric rows plus source metadata."""
    artifact_path = Path(path)
    data = _load_json(artifact_path)
    source: dict[str, Any] = {"path": str(artifact_path)}

    if isinstance(data, dict) and isinstance(data.get("metric_rows"), list):
        rows = _rows_from_row_dicts(
            data["metric_rows"],
            default_engine=engine,
            engine_override=engine,
        )
        source.update(
            {
                "kind": "row_set",
                "engines": _ordered_unique([row.engine for row in rows]),
            }
        )
        if data.get("gt_json"):
            source["gt_json"] = str(data["gt_json"])
        manifests = data.get("run_manifests")
        if isinstance(manifests, list):
            prediction_dirs = {}
            for manifest in manifests:
                if not isinstance(manifest, dict):
                    continue
                manifest_engine = manifest.get("engine")
                prediction_dir = manifest.get("prediction_dir")
                if manifest_engine and prediction_dir:
                    prediction_dirs[str(manifest_engine)] = str(prediction_dir)
            if prediction_dirs:
                source["prediction_dirs"] = prediction_dirs
        return rows, source

    if isinstance(data, dict):
        rows = []
        source["kind"] = "summary"
        for key in ("original_metric_rows", "postprocessed_metric_rows"):
            rows.extend(
                _rows_from_row_dicts(
                    data.get(key),
                    default_engine=engine,
                    engine_override=engine,
                )
            )
        if rows:
            source["engines"] = _ordered_unique([row.engine for row in rows])
            for key in (
                "gt_json",
                "source_run",
                "source_prediction_dir",
                "postprocessed_prediction_dir",
                "config_path",
            ):
                if data.get(key):
                    source[key] = str(data[key])
            return rows, source

    if engine is None:
        raise ValueError(
            f"{artifact_path} is not a row-set/summary artifact; pass it as ENGINE=PATH "
            "when it is a raw OmniDocBench metric-result JSON"
        )
    rows = OdlBenchAdapter.parse_metric_result(artifact_path, engine=engine)
    source.update({"kind": "metric_result", "engine": engine, "engines": [engine]})
    return rows, source


def infer_metric_result_path(
    prediction_dir: str | Path,
    *,
    result_dir: str | Path | None = None,
    match_method: str = "quick_match",
) -> Path:
    """Infer the raw OmniDocBench metric result path for an existing prediction dir."""
    if result_dir is None:
        bench_root = bootstrap.bench_root()
        if bench_root is None:
            raise RuntimeError("cannot infer result dir: opendataloader-bench root not found")
        result_root = bench_root / "result"
    else:
        result_root = Path(result_dir)
    save_name = OdlBenchAdapter.save_name_for(prediction_dir, match_method=match_method)
    return result_root / f"{save_name}_metric_result.json"


def rows_from_prediction_dir(
    engine: str,
    prediction_dir: str | Path,
    *,
    result_dir: str | Path | None = None,
    match_method: str = "quick_match",
) -> tuple[list[MetricRow], dict[str, Any]]:
    """Read the already-scored result associated with an existing prediction dir."""
    result_path = infer_metric_result_path(
        prediction_dir,
        result_dir=result_dir,
        match_method=match_method,
    )
    rows = OdlBenchAdapter.parse_metric_result(result_path, engine=engine)
    source = {
        "kind": "prediction_dir",
        "engine": engine,
        "engines": [engine],
        "prediction_dir": str(prediction_dir),
        "metric_result_path": str(result_path),
        "match_method": match_method,
    }
    return rows, source


def _source_refs_by_engine(sources: list[dict[str, Any]]) -> dict[str, list[int]]:
    refs: dict[str, list[int]] = {}
    for idx, source in enumerate(sources, start=1):
        for engine in source.get("engines") or []:
            refs.setdefault(str(engine), []).append(idx)
    return refs


def comparison_rows(metric_rows: list[MetricRow], sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pivot metric rows into one JSON-friendly comparison row per engine."""
    engines = _ordered_unique([row.engine for row in metric_rows])
    source_refs = _source_refs_by_engine(sources)
    table_rows: list[dict[str, Any]] = []
    for engine in engines:
        by_family: dict[str, MetricRow] = {}
        for row in metric_rows:
            if row.engine == engine and row.metric_family in METRIC_ORDER:
                by_family[row.metric_family] = row
        metrics: dict[str, dict[str, Any] | None] = {}
        for family in METRIC_ORDER:
            row = by_family.get(family)
            metrics[family] = row.to_dict() if row else None
        table_rows.append(
            {
                "engine": engine,
                "metrics": metrics,
                "source_refs": source_refs.get(engine, []),
            }
        )
    return table_rows


def build_existing_comparison(
    *,
    artifacts: list[tuple[str | None, Path]] | None = None,
    prediction_dirs: list[tuple[str | None, Path]] | None = None,
    result_dir: str | Path | None = None,
    gt_json: str | Path | None = None,
    match_method: str = "quick_match",
) -> dict[str, Any]:
    """Build a comparison payload from existing files only."""
    metric_rows: list[MetricRow] = []
    sources: list[dict[str, Any]] = []
    notes: list[str] = []
    inferred_gt = str(gt_json) if gt_json else ""

    for engine, artifact_path in artifacts or []:
        rows, source = rows_from_artifact(artifact_path, engine=engine)
        metric_rows.extend(rows)
        sources.append(source)
        inferred_gt = inferred_gt or str(source.get("gt_json") or "")

    for engine, prediction_dir in prediction_dirs or []:
        if engine is None:
            raise ValueError(f"prediction-dir specs require ENGINE=DIR: {prediction_dir}")
        rows, source = rows_from_prediction_dir(
            engine,
            prediction_dir,
            result_dir=result_dir,
            match_method=match_method,
        )
        metric_rows.extend(rows)
        sources.append(source)

    seen: set[tuple[str, str]] = set()
    for row in metric_rows:
        key = (row.engine, row.metric_family)
        if key in seen:
            notes.append(
                f"duplicate metric row for engine={row.engine} family={row.metric_family}; "
                "later rows win in the comparison table"
            )
        seen.add(key)

    payload = {
        "schema": SCHEMA,
        "gt_json": inferred_gt,
        "metric_directions": METRIC_DIRECTIONS,
        "metric_rows": [row.to_dict() for row in metric_rows],
        "comparison_rows": comparison_rows(metric_rows, sources),
        "sources": sources,
        "notes": notes,
    }
    return payload


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, (int, float)):
        return str(value)
    if abs(float(value)) >= 100:
        return f"{float(value):.2f}"
    return f"{float(value):.6g}"


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def render_markdown_table(payload: dict[str, Any]) -> str:
    """Render the comparison payload as a compact Markdown table."""
    headers = ["engine", *[METRIC_LABELS[family] for family in METRIC_ORDER], "sources"]
    lines = [
        "# Existing ODL/PaddleOCR Comparison",
        "",
        f"Schema: `{payload.get('schema', SCHEMA)}`",
    ]
    if payload.get("gt_json"):
        lines.append(f"GT: `{payload['gt_json']}`")
    lines.extend(["", "| " + " | ".join(headers) + " |"])
    aligns = ["---"] + ["---:"] * len(METRIC_ORDER) + ["---"]
    lines.append("| " + " | ".join(aligns) + " |")
    for row in payload.get("comparison_rows") or []:
        metrics = row.get("metrics") or {}
        cells = [_md_escape(row.get("engine") or "")]
        for family in METRIC_ORDER:
            metric = metrics.get(family) or {}
            cells.append(_format_value(metric.get("value")))
        cells.append(",".join(str(ref) for ref in row.get("source_refs") or []))
        lines.append("| " + " | ".join(cells) + " |")

    sources = payload.get("sources") or []
    if sources:
        lines.extend(["", "## Sources", ""])
        for idx, source in enumerate(sources, start=1):
            kind = source.get("kind", "artifact")
            path = source.get("path") or source.get("metric_result_path") or source.get("prediction_dir")
            engines = ",".join(str(e) for e in source.get("engines") or [])
            lines.append(f"{idx}. `{kind}` `{engines}` `{path}`")
    if payload.get("notes"):
        lines.extend(["", "## Notes", ""])
        for note in payload["notes"]:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def write_existing_comparison(
    payload: dict[str, Any],
    out_dir: str | Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """Write JSON and Markdown outputs, refusing overwrite unless requested."""
    out = Path(out_dir)
    json_path = out / JSON_NAME
    md_path = out / MARKDOWN_NAME
    existing = [path for path in (json_path, md_path) if path.exists()]
    if existing and not force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing comparison output(s): {names}")
    out.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_table(payload), encoding="utf-8")
    return json_path, md_path
