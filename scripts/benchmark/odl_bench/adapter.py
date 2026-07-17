"""OdlBenchAdapter — wire EPYC deterministic PDF backends into opendataloader-bench.

Wave-2 B3. Deterministic rows only; no inference; no commits.

Pipeline per engine:
  1. generate_predictions() — run a deterministic backend (pdftotext / ODL-local /
     LiteParse) over the source PDFs mapped to each GT page, writing one
     ``<gt_stem>.md`` prediction into a per-engine dir. Captures per-page latency
     (the SPEED rows — the harness does not time engines).
  2. emit_config() — write an OmniDocBench YAML config pointing prediction.data_path
     at that dir (metrics: text_block Edit_dist, table TEDS+Edit_dist, reading_order
     Edit_dist).
  3. score() — run ``pdf_validation.py --config`` under the BENCH venv (cwd=bench_root)
     and parse ``result/<save>_metric_result.json`` into structural/table/reading_order
     MetricRows.

score() is the only step that shells out; generate_predictions()/emit_config() are
pure + deterministic and are what the unit tests exercise. Model-gated engines are
NOT run here — see manifest_stubs.model_gated_stubs().
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from . import bootstrap, run_configs
from .backends import DETERMINISTIC_ENGINES, resolve_backend
from .manifest_stubs import model_gated_manifest, model_gated_stubs
from .paddleocr_vl import (
    DEFAULT_BINARY as PADDLEOCR_DEFAULT_BINARY,
    DEFAULT_MMPROJ as PADDLEOCR_DEFAULT_MMPROJ,
    DEFAULT_MODEL as PADDLEOCR_DEFAULT_MODEL,
    PADDLEOCR_VL_ENGINE,
    PROMPT_PROFILES as PADDLEOCR_PROMPT_PROFILES,
    PaddleOcrVlConfig,
    PaddleOcrVlProducer,
)
from .schemas import (
    METRIC_READING_ORDER,
    METRIC_STRUCTURAL,
    METRIC_TABLE,
    DeterministicRowSet,
    EngineRunManifest,
    MetricRow,
    PredictionArtifact,
)


class OdlBenchAdapter:
    def __init__(self, bench_root: str | Path | None = None):
        self.bench_root = Path(bench_root) if bench_root else bootstrap.bench_root()

    # ------------------------------------------------------------------ inputs
    @staticmethod
    def load_pdf_manifest(manifest_path: str | Path) -> dict[str, str]:
        """Load a GT-image -> source-PDF mapping.

        Accepts either ``{"<gt_image_basename>": "<pdf_path>", ...}`` or
        ``{"pairs": [{"gt_image": ..., "pdf": ...}, ...]}``.
        """
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "pairs" in data:
            return {row["gt_image"]: row["pdf"] for row in data["pairs"]}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        raise ValueError("pdf manifest must be a dict or {'pairs': [...]}")

    # ------------------------------------------------------- prediction phase
    def generate_predictions(
        self,
        engine: str,
        gt_json: str | Path,
        pdf_manifest: dict[str, str],
        out_dir: str | Path,
    ) -> EngineRunManifest:
        """Run ``engine`` over mapped PDFs, writing ``<stem>.md`` per GT page.

        GT pages with no mapped source PDF are skipped (recorded in ``detail``);
        this is deterministic and never calls a model.
        """
        backend = resolve_backend(engine)
        avail, reason = backend.available()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        gt_images = run_configs.gt_image_basenames(gt_json)
        artifacts: list[PredictionArtifact] = []
        skipped = 0
        for gt_image in gt_images:
            pred_name = run_configs.prediction_filename_for(gt_image)
            source_pdf = pdf_manifest.get(gt_image, "")
            if not source_pdf:
                skipped += 1
                continue
            if not avail:
                # Engine absent here: still emit an empty prediction so the row set
                # is complete + the harness sees a (zero-score) prediction rather
                # than a missing file. latency 0 marks "not measured".
                (out_dir / pred_name).write_text("", encoding="utf-8")
                artifacts.append(PredictionArtifact(gt_image, pred_name, source_pdf, 0, 0.0))
                continue
            outcome = backend.run(Path(source_pdf))
            (out_dir / pred_name).write_text(outcome.text, encoding="utf-8")
            artifacts.append(
                PredictionArtifact(
                    gt_image=gt_image,
                    prediction_filename=pred_name,
                    source_pdf=source_pdf,
                    char_count=outcome.char_count,
                    latency_ms=outcome.latency_ms,
                )
            )

        detail = "" if avail else f"engine unavailable in this interpreter: {reason}"
        if skipped:
            detail = (detail + "; " if detail else "") + f"{skipped} GT pages had no mapped PDF"
        return EngineRunManifest(
            engine=engine,
            kind="deterministic",
            available=avail,
            prediction_dir=str(out_dir),
            artifacts=artifacts,
            detail=detail,
        )

    # ------------------------------------------------------------ config phase
    def emit_config(
        self,
        prediction_dir: str | Path,
        gt_json: str | Path,
        out_config: str | Path,
    ) -> Path:
        config = run_configs.build_bench_config(str(prediction_dir), str(gt_json))
        out_config = Path(out_config)
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text(run_configs.dump_config_yaml(config), encoding="utf-8")
        return out_config

    # ------------------------------------------------------------ score phase
    @staticmethod
    def save_name_for(prediction_dir: str | Path, match_method: str = "quick_match") -> str:
        """Mirror ``pipeline.build_save_name`` for a config with a prediction path."""
        return os.path.basename(str(prediction_dir).rstrip("/")) + "_" + match_method

    def score_command(self, config_path: str | Path,
                      bench_python: str | Path | None = None) -> list[str]:
        """The exact scoring command (list form); does not execute it."""
        py = str(bench_python or bootstrap.bench_python() or run_configs.BENCH_PYTHON_DEFAULT)
        return [py, run_configs.BENCH_VALIDATION_SCRIPT, "--config", str(config_path)]

    def score(
        self,
        config_path: str | Path,
        prediction_dir: str | Path,
        bench_python: str | Path | None = None,
        timeout_sec: int = 1800,
    ) -> list[MetricRow]:
        """Run the harness (BENCH venv, cwd=bench_root) and parse metric rows.

        Raises if the bench root/venv are missing — scoring cannot run in the
        research venv (py3.14, no Levenshtein/apted).
        """
        if self.bench_root is None or not self.bench_root.exists():
            raise RuntimeError("opendataloader-bench root not found; cannot score")
        cmd = self.score_command(config_path, bench_python)
        subprocess.run(cmd, cwd=str(self.bench_root), check=True, timeout=timeout_sec)
        save_name = self.save_name_for(prediction_dir)
        result_path = self.bench_root / "result" / f"{save_name}_metric_result.json"
        engine = os.path.basename(str(prediction_dir).rstrip("/"))
        return self.parse_metric_result(result_path, engine)

    # --------------------------------------------------------- result parsing
    @staticmethod
    def parse_metric_result(result_path: str | Path, engine: str) -> list[MetricRow]:
        """Map ``<save>_metric_result.json`` -> structural/table/reading_order rows.

        Known OmniDocBench nesting (verified against the demo result):
          text_block.all.Edit_dist.ALL_page_avg   -> structural (lower=better)
          table.all.TEDS.all                       -> table (higher=better)
          table.all.TEDS_structure_only.all        -> table structure-only (secondary)
          reading_order.all.Edit_dist.ALL_page_avg -> reading order (lower=better)
        """
        with open(result_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        def leaf(*keys):
            cur = data
            for k in keys:
                if not isinstance(cur, dict) or k not in cur:
                    return None
                cur = cur[k]
            return cur if isinstance(cur, (int, float)) else None

        rows: list[MetricRow] = []
        rows.append(MetricRow(
            engine=engine,
            metric_family=METRIC_STRUCTURAL,
            metric_name="text_block.Edit_dist.ALL_page_avg",
            value=leaf("text_block", "all", "Edit_dist", "ALL_page_avg"),
            detail="normalized edit distance; LOWER is better",
        ))
        teds = leaf("table", "all", "TEDS", "all")
        teds_struct = leaf("table", "all", "TEDS_structure_only", "all")
        rows.append(MetricRow(
            engine=engine,
            metric_family=METRIC_TABLE,
            metric_name="table.TEDS.all",
            value=teds,
            detail=(
                "TEDS; HIGHER is better"
                + (f"; TEDS_structure_only={teds_struct}" if teds_struct is not None else "")
            ),
        ))
        rows.append(MetricRow(
            engine=engine,
            metric_family=METRIC_READING_ORDER,
            metric_name="reading_order.Edit_dist.ALL_page_avg",
            value=leaf("reading_order", "all", "Edit_dist", "ALL_page_avg"),
            detail="normalized edit distance; LOWER is better",
        ))
        return rows

    # -------------------------------------------------- full deterministic set
    def build_deterministic_row_set(
        self,
        gt_json: str | Path,
        pdf_manifest: dict[str, str],
        run_dir: str | Path,
        engines: tuple[str, ...] = DETERMINISTIC_ENGINES,
        do_score: bool = False,
        bench_python: str | Path | None = None,
    ) -> DeterministicRowSet:
        """Wire the deterministic comparison for all engines.

        With ``do_score=False`` (default; safe under the research venv) only the
        prediction + speed rows are produced; structural/table/reading_order rows
        are added when ``do_score=True`` (BENCH venv, real fixtures).
        """
        run_dir = Path(run_dir)
        row_set = DeterministicRowSet(engines=list(engines), gt_json=str(gt_json))
        for engine in engines:
            pred_dir = run_dir / "predictions" / engine
            manifest = self.generate_predictions(engine, gt_json, pdf_manifest, pred_dir)
            row_set.run_manifests.append(manifest)
            row_set.metric_rows.append(manifest.speed_row())
            if manifest.detail:
                row_set.notes.append(f"{engine}: {manifest.detail}")
            if do_score:
                cfg = self.emit_config(pred_dir, gt_json, run_dir / "config" / f"{engine}.yaml")
                row_set.metric_rows.extend(self.score(cfg, pred_dir, bench_python))
        return row_set

    # --------------------------------------------------------- model-gated run
    def generate_model_gated_predictions(
        self,
        engine: str,
        gt_json: str | Path,
        out_dir: str | Path,
        *,
        image_root: str | Path | None = None,
        response_dir: str | Path | None = None,
        allow_inference: bool = False,
        paddle_config: PaddleOcrVlConfig | None = None,
    ) -> EngineRunManifest:
        if engine != PADDLEOCR_VL_ENGINE:
            raise ValueError(f"unknown model-gated engine {engine!r}; known: {(PADDLEOCR_VL_ENGINE,)}")
        if not allow_inference:
            raise PermissionError("model-gated producers require --allow-inference")
        out_dir = Path(out_dir)
        response_dir = Path(response_dir) if response_dir else out_dir.parent / f"{out_dir.name}_responses"
        producer = PaddleOcrVlProducer(paddle_config or PaddleOcrVlConfig())
        return producer.generate(
            gt_json=gt_json,
            image_root=image_root,
            prediction_dir=out_dir,
            response_dir=response_dir,
        )

    def build_model_gated_row_set(
        self,
        gt_json: str | Path,
        run_dir: str | Path,
        *,
        engine: str,
        image_root: str | Path | None = None,
        allow_inference: bool = False,
        paddle_config: PaddleOcrVlConfig | None = None,
        do_score: bool = False,
        bench_python: str | Path | None = None,
    ) -> DeterministicRowSet:
        run_dir = Path(run_dir)
        row_set = DeterministicRowSet(engines=[engine], gt_json=str(gt_json))
        pred_dir = run_dir / "predictions" / engine
        response_dir = run_dir / "responses" / engine
        manifest = self.generate_model_gated_predictions(
            engine,
            gt_json,
            pred_dir,
            image_root=image_root,
            response_dir=response_dir,
            allow_inference=allow_inference,
            paddle_config=paddle_config,
        )
        row_set.run_manifests.append(manifest)
        row_set.metric_rows.append(manifest.speed_row())
        if manifest.detail:
            row_set.notes.append(f"{engine}: {manifest.detail}")
        if do_score:
            cfg = self.emit_config(pred_dir, gt_json, run_dir / "config" / f"{engine}.yaml")
            row_set.metric_rows.extend(self.score(cfg, pred_dir, bench_python))
        return row_set

    # ------------------------------------------------------- model-gated stubs
    @staticmethod
    def model_gated_manifest_stubs():
        return model_gated_stubs()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="ODL structural bench adapter (Wave-2 B3)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_av = sub.add_parser("availability", help="report deterministic engine availability here")

    p_stub = sub.add_parser("stubs", help="print model-gated Wave-3 manifest stubs (JSON)")

    p_run = sub.add_parser("run", help="generate predictions (+optional score) for engines")
    p_run.add_argument("--engine", action="append", help="engine(s); default = all deterministic")
    p_run.add_argument("--gt", required=True, help="OmniDocBench GT json")
    p_run.add_argument("--pdf-manifest", required=True, help="GT-image -> source-PDF json")
    p_run.add_argument("--run-dir", required=True)
    p_run.add_argument("--score", action="store_true", help="also run harness scoring (bench venv)")
    p_run.add_argument("--bench-python", default=None)

    p_model = sub.add_parser("run-model", help="run an explicit model-gated prediction producer")
    p_model.add_argument("--engine", required=True, choices=[PADDLEOCR_VL_ENGINE])
    p_model.add_argument("--gt", required=True, help="OmniDocBench GT json")
    p_model.add_argument("--image-root", default=None, help="directory containing GT page images")
    p_model.add_argument("--run-dir", required=True)
    p_model.add_argument("--score", action="store_true", help="also run harness scoring (bench venv)")
    p_model.add_argument("--bench-python", default=None)
    p_model.add_argument("--allow-inference", action="store_true")
    p_model.add_argument("--binary", type=Path, default=PADDLEOCR_DEFAULT_BINARY)
    p_model.add_argument("--model", type=Path, default=PADDLEOCR_DEFAULT_MODEL)
    p_model.add_argument("--mmproj", type=Path, default=PADDLEOCR_DEFAULT_MMPROJ)
    p_model.add_argument("--port", type=int, default=19330)
    p_model.add_argument("--context", type=int, default=8192)
    p_model.add_argument("--threads", type=int, default=24)
    p_model.add_argument("--parallel", type=int, default=1)
    p_model.add_argument("--device", default="ROCm0")
    p_model.add_argument("--gpu-layers", type=int, default=99)
    p_model.add_argument("--max-tokens", type=int, default=2048)
    p_model.add_argument("--startup-timeout", type=int, default=240)
    p_model.add_argument("--request-timeout", type=int, default=900)
    p_model.add_argument("--allow-dirty-host", action="store_true")
    p_model.add_argument(
        "--prompt-profile",
        choices=sorted(PADDLEOCR_PROMPT_PROFILES),
        default="default",
        help="PaddleOCR-VL extraction prompt profile",
    )
    p_model.add_argument("--prompt-file", type=Path, default=None, help="override prompt text from file")

    args = parser.parse_args(argv)

    if args.cmd == "availability":
        from .backends import availability_report

        print(json.dumps(availability_report(), indent=2))
        return 0
    if args.cmd == "stubs":
        print(json.dumps(model_gated_manifest(), indent=2))
        return 0
    if args.cmd == "run":
        adapter = OdlBenchAdapter()
        pdf_manifest = OdlBenchAdapter.load_pdf_manifest(args.pdf_manifest)
        engines = tuple(args.engine) if args.engine else DETERMINISTIC_ENGINES
        row_set = adapter.build_deterministic_row_set(
            args.gt, pdf_manifest, args.run_dir, engines=engines,
            do_score=args.score, bench_python=args.bench_python,
        )
        out = Path(args.run_dir) / "deterministic_row_set.json"
        out.write_text(json.dumps(row_set.to_dict(), indent=2), encoding="utf-8")
        print(f"[odl_bench] wrote {out}")
        print(json.dumps([r.to_dict() for r in row_set.metric_rows], indent=2))
        return 0
    if args.cmd == "run-model":
        if not args.allow_inference:
            parser.error("run-model requires --allow-inference")
        adapter = OdlBenchAdapter()
        prompt = (
            args.prompt_file.read_text(encoding="utf-8")
            if args.prompt_file
            else PADDLEOCR_PROMPT_PROFILES[args.prompt_profile]
        )
        cfg = PaddleOcrVlConfig(
            binary=args.binary,
            model=args.model,
            mmproj=args.mmproj,
            port=args.port,
            context=args.context,
            threads=args.threads,
            parallel=args.parallel,
            device=args.device,
            gpu_layers=args.gpu_layers,
            max_tokens=args.max_tokens,
            startup_timeout_s=args.startup_timeout,
            request_timeout_s=args.request_timeout,
            prompt=prompt,
            prompt_profile=str(args.prompt_file) if args.prompt_file else args.prompt_profile,
            allow_dirty_host=args.allow_dirty_host,
        )
        row_set = adapter.build_model_gated_row_set(
            args.gt,
            args.run_dir,
            engine=args.engine,
            image_root=args.image_root,
            allow_inference=args.allow_inference,
            paddle_config=cfg,
            do_score=args.score,
            bench_python=args.bench_python,
        )
        out = Path(args.run_dir) / "model_gated_row_set.json"
        out.write_text(json.dumps(row_set.to_dict(), indent=2), encoding="utf-8")
        print(f"[odl_bench] wrote {out}")
        print(json.dumps([r.to_dict() for r in row_set.metric_rows], indent=2))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
