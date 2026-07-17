"""Row + manifest schemas for the ODL structural bench wiring (Wave-2 B3).

Two families of records:

* Deterministic run records — produced NOW by running pdftotext / ODL-local /
  LiteParse over source PDFs and scoring against OmniDocBench GT:
    - ``ExtractionOutcome``   : one engine's extraction of one PDF (text+latency)
    - ``PredictionArtifact``  : the ``<stem>.md`` file written for one GT page
    - ``EngineRunManifest``   : all predictions + speed aggregate for one engine
    - ``MetricRow``           : one (engine, metric_family) score row
    - ``DeterministicRowSet`` : the full pdftotext-vs-ODL-vs-LiteParse comparison

* ``ManifestEntryStub`` — a model-gated engine (LightOnOCR / VL / ODL-hybrid)
  emitted as a Wave-3 batch-manifest entry INSTEAD of being run here (no inference
  in this wave). It is a structured spec of what to run, its preconditions, and
  its expected artifacts.

All records are plain dataclasses with ``to_dict()`` for JSON serialisation.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


# Metric families this bench reports (mapped onto OmniDocBench metric registry
# names inside run_configs.DETERMINISTIC_METRIC_CONFIG).
METRIC_STRUCTURAL = "structural_fidelity"   # text_block Edit_dist
METRIC_TABLE = "table_fidelity"             # table TEDS (+ Edit_dist)
METRIC_READING_ORDER = "reading_order"      # reading_order Edit_dist
METRIC_SPEED = "speed"                      # our own extraction latency_ms

DETERMINISTIC_KIND = "deterministic"
MODEL_GATED_KIND = "model_gated"


@dataclass
class ExtractionOutcome:
    """Result of running one engine on one PDF (no scoring yet)."""

    engine: str
    pdf_path: str
    available: bool          # engine importable/runnable in this interpreter
    ok: bool                 # produced non-empty text
    text: str
    latency_ms: float
    method: str = ""         # engine's own method label (e.g. "pdftotext")
    char_count: int = 0
    detail: str = ""         # unavailability / failure reason

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Text can be large; expose only a length + a short head in serialised form.
        d.pop("text", None)
        d["char_count"] = len(self.text)
        d["text_head"] = self.text[:120]
        return d


@dataclass
class PredictionArtifact:
    """One prediction markdown file written for one GT page."""

    gt_image: str            # GT page_info.image_path basename (e.g. foo.pdf_7.jpg)
    prediction_filename: str  # what the bench looks for (foo.pdf_7.md)
    source_pdf: str          # PDF the engine read (may be "" for image producers)
    char_count: int
    latency_ms: float
    source_image: str = ""   # page image read by a model-gated producer
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prompt_tps: float | None = None
    decode_tps: float | None = None
    finish_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineRunManifest:
    """All predictions + the speed aggregate for a single engine."""

    engine: str
    kind: str                # DETERMINISTIC_KIND
    available: bool
    prediction_dir: str
    artifacts: list[PredictionArtifact] = field(default_factory=list)
    detail: str = ""

    def latencies(self) -> list[float]:
        return [a.latency_ms for a in self.artifacts if a.latency_ms > 0]

    def speed_row(self) -> "MetricRow":
        lat = self.latencies()
        value = statistics.median(lat) if lat else None
        return MetricRow(
            engine=self.engine,
            metric_family=METRIC_SPEED,
            metric_name="latency_ms_median",
            value=value,
            n=len(self.artifacts),
            detail=(
                "median per-page extraction latency (ms); "
                f"mean={statistics.mean(lat):.3f}" if lat else "no timed pages"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "kind": self.kind,
            "available": self.available,
            "prediction_dir": self.prediction_dir,
            "detail": self.detail,
            "n_artifacts": len(self.artifacts),
            "artifacts": [a.to_dict() for a in self.artifacts],
        }


@dataclass
class MetricRow:
    """One (engine, metric-family) score row."""

    engine: str
    metric_family: str       # METRIC_STRUCTURAL / _TABLE / _READING_ORDER / _SPEED
    metric_name: str         # e.g. "Edit_dist", "TEDS", "latency_ms_median"
    value: float | None      # None => not-yet-scored / unavailable
    n: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeterministicRowSet:
    """The wired deterministic comparison: engines x {structural,table,order,speed}."""

    engines: list[str]
    gt_json: str
    metric_rows: list[MetricRow] = field(default_factory=list)
    run_manifests: list[EngineRunManifest] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engines": self.engines,
            "gt_json": self.gt_json,
            "metric_rows": [r.to_dict() for r in self.metric_rows],
            "run_manifests": [m.to_dict() for m in self.run_manifests],
            "notes": self.notes,
        }


@dataclass
class ManifestEntryStub:
    """A model-gated engine emitted as a Wave-3 batch-manifest entry (not run now).

    Consumed by the operator's inference-batch loop. Deliberately structured so a
    downstream compiler can turn it straight into an execution_manifest row.
    """

    entry_id: str
    engine: str
    kind: str = MODEL_GATED_KIND
    description: str = ""
    preconditions: list[str] = field(default_factory=list)
    command: str = ""                 # the exact command to run when unblocked
    env: dict[str, str] = field(default_factory=dict)
    expected_artifacts: list[str] = field(default_factory=list)
    reuses_deterministic_wiring: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
