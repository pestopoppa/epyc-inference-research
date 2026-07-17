"""Canonical run constants + OmniDocBench config template for deterministic rows.

OmniDocBench-bench has no in-process "engine plugin" API. Its registries
(``DATASET_REGISTRY`` / ``METRIC_REGISTRY`` / ``EVAL_TASK_REGISTRY``) register
datasets/metrics/tasks, NOT engines. An "engine" is an EXTERNAL prediction
producer: it writes one ``<image_stem>.md`` per GT page into a prediction dir,
then the harness scores that dir against the GT JSON via a YAML config
(``python pdf_validation.py --config <cfg>``).

So "registering our engine" == (1) generate a bench-format prediction dir with an
odl_bench backend, (2) emit a config pointing ``prediction.data_path`` at it,
(3) run the harness under the BENCH venv. This module holds the config template
and the deterministic metric selection.

Metric-family -> OmniDocBench registry-name mapping (deterministic only; CDM is
intentionally omitted — it needs TeX Live / Ghostscript / ImageMagick and is not
part of the structural/table/reading-order/speed row set):

    structural_fidelity -> text_block:     Edit_dist
    table_fidelity       -> table:          TEDS, Edit_dist
    reading_order        -> reading_order:  Edit_dist
    speed                -> (measured by odl_bench at extraction time, not by the harness)
"""

from __future__ import annotations

from pathlib import Path

# Interpreter that MUST run scoring (has Levenshtein/apted; py3.11).
# Resolved lazily via bootstrap.bench_python(); this string documents the default.
BENCH_PYTHON_DEFAULT = "/mnt/raid0/llm/opendataloader-bench/.venv/bin/python"
BENCH_VALIDATION_SCRIPT = "pdf_validation.py"  # bench entrypoint, run with cwd=bench_root

# Interpreter recommended to GENERATE predictions with all deterministic engines
# (has opendataloader_pdf). pdftotext works from any interpreter.
ODL_CAPABLE_PYTHON = "/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python"

# Metric config block (mirrors configs/end2end.yaml, minus CDM). Worker counts are
# conservative to avoid oversubscribing the shared EPYC host.
DETERMINISTIC_METRIC_CONFIG = {
    "text_block": {"metric": ["Edit_dist"]},
    "table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": 8},
    "reading_order": {"metric": ["Edit_dist"]},
}


def build_bench_config(prediction_dir: str, gt_json: str,
                       match_workers: int = 8) -> dict:
    """Return an OmniDocBench end2end config dict for a deterministic engine run.

    Structure matches ``configs/end2end.yaml`` so ``pipeline.run_config`` accepts it
    unchanged. Paths are absolute so the config is cwd-independent for callers,
    though the bench itself still needs cwd=bench_root for its ``./result`` output.
    """
    return {
        "end2end_eval": {
            "metrics": DETERMINISTIC_METRIC_CONFIG,
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(gt_json)},
                "prediction": {"data_path": str(prediction_dir)},
                "match_method": "quick_match",
                "match_workers": match_workers,
                "quick_match_truncated_timeout_sec": 300,
                "match_timeout_sec": 420,
            },
        }
    }


def dump_config_yaml(config: dict) -> str:
    """Serialise a config dict to YAML text (pyyaml if present, else a tiny writer)."""
    try:
        import yaml  # research + bench venvs both ship pyyaml

        return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    except Exception:  # pragma: no cover - fallback path
        return _minimal_yaml(config)


def _minimal_yaml(obj, indent: int = 0) -> str:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.append(_minimal_yaml(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_minimal_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}- {item}")
    else:
        return f"{pad}{obj}"
    return "\n".join(x for x in lines if x)


def gt_image_basenames(gt_json: str | Path) -> list[str]:
    """Read the GT JSON and return each page's ``page_info.image_path`` basename."""
    import json
    import os

    with open(gt_json, "r", encoding="utf-8") as fh:
        pages = json.load(fh)
    names = []
    for page in pages:
        img = page.get("page_info", {}).get("image_path", "")
        if img:
            names.append(os.path.basename(img))
    return names


def gt_image_paths(gt_json: str | Path, image_root: str | Path | None = None) -> dict[str, Path]:
    """Resolve each GT image basename to the page image path a VL producer should read.

    OmniDocBench GT often stores a basename in ``page_info.image_path`` and keeps
    files under ``<gt_dir>/images``. Some derived corpora use absolute or relative
    paths. This resolver preserves the basename contract used for prediction
    filenames while making the image-source root explicit for model-gated arms.
    Missing files are still returned as best-effort candidates so the caller can
    record a complete skip/error manifest instead of silently dropping the page.
    """
    import json
    import os

    gt_path = Path(gt_json)
    root = Path(image_root) if image_root else None
    with gt_path.open("r", encoding="utf-8") as fh:
        pages = json.load(fh)

    resolved: dict[str, Path] = {}
    for page in pages:
        raw = page.get("page_info", {}).get("image_path", "")
        if not raw:
            continue
        raw_path = Path(raw)
        basename = os.path.basename(raw)
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        if root is not None:
            candidates.extend([root / raw, root / basename])
        candidates.extend([gt_path.parent / raw, gt_path.parent / "images" / basename])
        chosen = next((path for path in candidates if path.exists()), candidates[0])
        resolved[basename] = chosen
    return resolved


def prediction_filename_for(gt_image_basename: str) -> str:
    """Prediction filename the harness resolves for a GT page.

    Mirrors ``End2EndDataset._resolve_prediction_path`` primary form:
    strip the 4-char image extension, append ``.md``.
    """
    return gt_image_basename[:-4] + ".md"
