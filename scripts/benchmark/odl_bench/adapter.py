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
from .intrinsic import (
    DefaultChunker,
    Embedder,
    score_prediction_dir,
)
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
from .unlimited_ocr import (
    DEFAULT_BINARY as UNLIMITED_OCR_DEFAULT_BINARY,
    DEFAULT_MMPROJ as UNLIMITED_OCR_DEFAULT_MMPROJ,
    DEFAULT_MODEL as UNLIMITED_OCR_DEFAULT_MODEL,
    UNLIMITED_OCR_ENGINE,
    PROMPT_PROFILES as UNLIMITED_OCR_PROMPT_PROFILES,
    UnlimitedOcrConfig,
    UnlimitedOcrProducer,
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

    # ---------------------------------------------------- intrinsic scoring
    @staticmethod
    def score_intrinsic(
        prediction_dir: str | Path,
        engine: str,
        *,
        embedder: Embedder | None = None,
        chunker: DefaultChunker | None = None,
        min_tokens: int = 100,
        max_tokens: int = 1100,
    ) -> list[MetricRow]:
        """Score one engine's prediction dir with Ekimetrics intrinsic metrics.

        Phase-3 contract (handoff :539-540): informational alongside
        NID/TEDS/MHS — never a gate on its own; FMRE/RC excluded. ICC/DCC rows
        carry ``value=None`` with the reason when ``embedder`` is None (the
        same degrade convention as a missing extraction backend).
        """
        return score_prediction_dir(
            prediction_dir,
            engine=engine,
            embedder=embedder,
            chunker=chunker,
        )

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
        unlimited_config: UnlimitedOcrConfig | None = None,
    ) -> EngineRunManifest:
        if engine not in (PADDLEOCR_VL_ENGINE, UNLIMITED_OCR_ENGINE):
            raise ValueError(
                f"unknown model-gated engine {engine!r}; known: "
                f"{sorted((PADDLEOCR_VL_ENGINE, UNLIMITED_OCR_ENGINE))}"
            )
        if not allow_inference:
            raise PermissionError("model-gated producers require --allow-inference")
        out_dir = Path(out_dir)
        response_dir = Path(response_dir) if response_dir else out_dir.parent / f"{out_dir.name}_responses"
        if engine == PADDLEOCR_VL_ENGINE:
            producer = PaddleOcrVlProducer(paddle_config or PaddleOcrVlConfig())
        else:
            producer = UnlimitedOcrProducer(unlimited_config or UnlimitedOcrConfig())
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
        unlimited_config: UnlimitedOcrConfig | None = None,
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
            unlimited_config=unlimited_config,
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

    sub.add_parser("availability", help="report deterministic engine availability here")

    sub.add_parser("stubs", help="print model-gated Wave-3 manifest stubs (JSON)")

    p_run = sub.add_parser("run", help="generate predictions (+optional score) for engines")
    p_run.add_argument("--engine", action="append", help="engine(s); default = all deterministic")
    p_run.add_argument("--gt", required=True, help="OmniDocBench GT json")
    p_run.add_argument("--pdf-manifest", required=True, help="GT-image -> source-PDF json")
    p_run.add_argument("--run-dir", required=True)
    p_run.add_argument("--score", action="store_true", help="also run harness scoring (bench venv)")
    p_run.add_argument("--bench-python", default=None)

    p_model = sub.add_parser("run-model", help="run an explicit model-gated prediction producer")
    p_model.add_argument(
        "--engine",
        required=True,
        choices=[PADDLEOCR_VL_ENGINE, UNLIMITED_OCR_ENGINE],
    )
    p_model.add_argument("--gt", required=True, help="OmniDocBench GT json")
    p_model.add_argument("--image-root", default=None, help="directory containing GT page images")
    p_model.add_argument("--run-dir", required=True)
    p_model.add_argument("--score", action="store_true", help="also run harness scoring (bench venv)")
    p_model.add_argument("--bench-python", default=None)
    p_model.add_argument("--allow-inference", action="store_true")
    # Per-engine defaults (binary/model/mmproj paths, port, max_tokens differ between
    # the PaddleOCR-VL and Unlimited-OCR lanes) are resolved in the handler below;
    # argparse defaults stay None so the chosen engine's module defaults apply.
    p_model.add_argument("--binary", type=Path, default=None, help="llama-server binary (per-engine default)")
    p_model.add_argument("--model", type=Path, default=None, help="GGUF model (per-engine default)")
    p_model.add_argument("--mmproj", type=Path, default=None, help="vision mmproj (per-engine default)")
    p_model.add_argument("--port", type=int, default=None, help="server port (per-engine default)")
    p_model.add_argument("--context", type=int, default=8192)
    p_model.add_argument("--threads", type=int, default=24)
    p_model.add_argument("--parallel", type=int, default=1)
    p_model.add_argument("--device", default="ROCm0")
    p_model.add_argument("--gpu-layers", type=int, default=99)
    p_model.add_argument("--max-tokens", type=int, default=None, help="per-engine default")
    p_model.add_argument("--startup-timeout", type=int, default=240)
    p_model.add_argument("--request-timeout", type=int, default=900)
    p_model.add_argument("--allow-dirty-host", action="store_true")
    p_model.add_argument(
        "--prompt-profile",
        choices=sorted(UNLIMITED_OCR_PROMPT_PROFILES),
        default="default",
        help="model-gated extraction prompt profile (PaddleOCR-VL or Unlimited-OCR)",
    )
    p_model.add_argument("--prompt-file", type=Path, default=None, help="override prompt text from file")

    p_compare = sub.add_parser(
        "compare-existing",
        help="materialize JSON/Markdown comparison from existing scored artifacts only",
    )

    p_intrinsic = sub.add_parser(
        "intrinsic",
        help="score an existing prediction dir with Ekimetrics intrinsic metrics "
        "(SC/BI always; ICC/DCC when an embedder is available)",
    )
    p_intrinsic.add_argument("--prediction-dir", required=True, help="dir of <stem>.md predictions")
    p_intrinsic.add_argument("--engine", required=True, help="engine label for the rows")
    p_intrinsic.add_argument(
        "--embedder",
        action="store_true",
        help="resolve a sentence-transformers embedder for ICC/DCC (absent -> value=None rows)",
    )
    p_intrinsic.add_argument("--chunk-size", type=int, default=600)
    p_intrinsic.add_argument("--min-tokens", type=int, default=100)
    p_intrinsic.add_argument("--max-tokens", type=int, default=1100)
    p_intrinsic.add_argument("--out", default=None, help="JSON path for the intrinsic row set")
    p_compare.add_argument(
        "--artifact",
        action="append",
        default=[],
        help=(
            "existing row-set/summary JSON path, ENGINE=artifact alias, or "
            "ENGINE=raw_metric_result.json; "
            "may be repeated"
        ),
    )
    p_compare.add_argument(
        "--prediction-dir",
        action="append",
        default=[],
        help=(
            "ENGINE=prediction_dir; infers the existing metric result from --result-dir "
            "using the quick_match save-name convention"
        ),
    )
    p_compare.add_argument(
        "--result-dir",
        default=None,
        help="directory containing existing *_metric_result.json files; default = bench result dir",
    )
    p_compare.add_argument("--gt", default=None, help="optional GT JSON path label for output")
    p_compare.add_argument("--out-dir", required=True, help="directory for comparison JSON/Markdown")
    p_compare.add_argument("--match-method", default="quick_match")
    p_compare.add_argument("--force", action="store_true", help="overwrite existing comparison outputs")

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
    if args.cmd == "compare-existing":
        from .comparison import (
            build_existing_comparison,
            parse_engine_path_specs,
            write_existing_comparison,
        )

        if not args.artifact and not args.prediction_dir:
            parser.error("compare-existing requires at least one --artifact or --prediction-dir")
        payload = build_existing_comparison(
            artifacts=parse_engine_path_specs(args.artifact),
            prediction_dirs=parse_engine_path_specs(args.prediction_dir),
            result_dir=args.result_dir,
            gt_json=args.gt,
            match_method=args.match_method,
        )
        json_path, md_path = write_existing_comparison(payload, args.out_dir, force=args.force)
        print(f"[odl_bench] wrote {json_path}")
        print(f"[odl_bench] wrote {md_path}")
        print(json.dumps(payload["comparison_rows"], indent=2))
        return 0
    if args.cmd == "intrinsic":
        adapter = OdlBenchAdapter()
        embedder = None
        if args.embedder:
            from .intrinsic import resolve_embedder

            embedder, reason = resolve_embedder()
            if embedder is None:
                print(f"[odl_bench] embedder unavailable: {reason} (ICC/DCC rows will be None)")
        rows = adapter.score_intrinsic(
            args.prediction_dir,
            args.engine,
            embedder=embedder,
            chunker=DefaultChunker(chunk_size=args.chunk_size),
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        payload = {
            "schema": "odl_bench.intrinsic.v1",
            "prediction_dir": str(args.prediction_dir),
            "engine": args.engine,
            "metric_rows": [r.to_dict() for r in rows],
        }
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[odl_bench] wrote {out}")
        print(json.dumps(payload["metric_rows"], indent=2))
        return 0
    if args.cmd == "run-model":
        if not args.allow_inference:
            parser.error("run-model requires --allow-inference")
        adapter = OdlBenchAdapter()
        if args.engine == UNLIMITED_OCR_ENGINE:
            cfg_cls, prompt_profiles = UnlimitedOcrConfig, UNLIMITED_OCR_PROMPT_PROFILES
            def_binary, def_model, def_mmproj = (
                UNLIMITED_OCR_DEFAULT_BINARY,
                UNLIMITED_OCR_DEFAULT_MODEL,
                UNLIMITED_OCR_DEFAULT_MMPROJ,
            )
            def_port, def_max_tokens = 19331, 4096
        else:
            cfg_cls, prompt_profiles = PaddleOcrVlConfig, PADDLEOCR_PROMPT_PROFILES
            def_binary, def_model, def_mmproj = (
                PADDLEOCR_DEFAULT_BINARY,
                PADDLEOCR_DEFAULT_MODEL,
                PADDLEOCR_DEFAULT_MMPROJ,
            )
            def_port, def_max_tokens = 19330, 2048
        prompt = (
            args.prompt_file.read_text(encoding="utf-8")
            if args.prompt_file
            else prompt_profiles[args.prompt_profile]
        )
        cfg = cfg_cls(
            binary=args.binary if args.binary is not None else def_binary,
            model=args.model if args.model is not None else def_model,
            mmproj=args.mmproj if args.mmproj is not None else def_mmproj,
            port=args.port if args.port is not None else def_port,
            context=args.context,
            threads=args.threads,
            parallel=args.parallel,
            device=args.device,
            gpu_layers=args.gpu_layers,
            max_tokens=args.max_tokens if args.max_tokens is not None else def_max_tokens,
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
            paddle_config=cfg if args.engine == PADDLEOCR_VL_ENGINE else None,
            unlimited_config=cfg if args.engine == UNLIMITED_OCR_ENGINE else None,
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
