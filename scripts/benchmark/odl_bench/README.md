# odl_bench — EPYC PDF backends × opendataloader-bench (Wave-2 B3)

Wire the EPYC PDF-extraction and opt-in model-gated producers into the
**opendataloader-bench** (OmniDocBench) harness. The deterministic comparison is
**pdftotext vs ODL-local vs LiteParse** on structural-fidelity, table-fidelity,
reading-order, and speed. Wave-3 model-gated engines reuse the same
`<gt_image_stem>.md` prediction/scoring path, but they only run behind explicit
`--allow-inference`.

Default commands remain **no inference**. ODL-local is rule-based XY-Cut++, not ML.

All files here are NEW (research-repo-owned). Nothing under `scripts/benchmark/`
outside this dir is edited, and the orchestrator `pdf_router` is **imported
read-only**, never modified.

## How the harness actually works (important)

opendataloader-bench is a **scoring harness**, not an engine runner. Its
registries (`DATASET_REGISTRY` / `METRIC_REGISTRY` / `EVAL_TASK_REGISTRY`) register
datasets/metrics/tasks — **there is no in-process "engine plugin" API.** An
"engine" is an **external prediction producer**: it writes one
`<gt_image_stem>.md` per GT page into a prediction directory, then the harness
scores that directory against the GT JSON via a YAML config:

```
<bench_root>/.venv/bin/python pdf_validation.py --config <cfg.yaml>   # cwd = bench_root
```

So **registering our engine = (1) generate a bench-format prediction dir with an
odl_bench backend, (2) emit a config pointing `prediction.data_path` at it,
(3) run the harness under the bench venv.** `emit_config()` can also drop the YAML
straight into the bench's `configs/` dir if you want it version-registered there.

**Prediction filename contract** (from `End2EndDataset._resolve_prediction_path`):
strip the 4-char image extension, append `.md` → `foo.pdf_7.jpg` ⇒ `foo.pdf_7.md`.

## Deterministic row set wired

| metric family | OmniDocBench source | direction | produced by |
|---|---|---|---|
| `structural_fidelity` | `text_block` → `Edit_dist.ALL_page_avg` | lower better | harness scoring |
| `table_fidelity` | `table` → `TEDS.all` (+ `TEDS_structure_only`, `Edit_dist`) | higher better | harness scoring |
| `reading_order` | `reading_order` → `Edit_dist.ALL_page_avg` | lower better | harness scoring |
| `speed` | per-page extraction `latency_ms` (median) | lower better | **odl_bench** (backends time themselves; the harness does not time engines) |
| `intrinsic_chunk_quality` | Ekimetrics SC/BI (+ ICC/DCC with an embedder) | higher better | **odl_bench** (`intrinsic` scoring; informational, never a gate) |

CDM/display_formula are intentionally omitted (CDM needs TeX Live / Ghostscript /
ImageMagick; out of scope for the structural/table/order/speed set).

## Intrinsic chunk-quality scoring (ODL-011)

The bench scores extraction fidelity (NID/TEDS/MHS); the intrinsic rows score
the *chunking quality* of each engine's extracted markdown using the Ekimetrics
MIT scaffold (intake-579/580) — the instrumented harness for the Phase-2
Ekimetrics-vs-HOPE side-by-side contract (`handoffs/active/
opendataloader-pipeline-integration.md` :539-540).

```bash
# score an existing prediction dir (no inference, no model download)
$RES/.venv/bin/python -m scripts.benchmark.odl_bench.adapter intrinsic \
  --prediction-dir <pred_dir> --engine pdftotext --out /tmp/intrinsic.json
```

Four metric rows are emitted per engine, all `higher_better`:

* `Ekimetrics.SC` — size compliance: fraction of chunks within token bounds.
* `Ekimetrics.BI` — block integrity: fraction of structural blocks (headings /
  paragraphs) not cut in half by the chunking.
* `Ekimetrics.ICC` — intrachunk cohesion: mean sentence-vs-chunk embedding
  cosine (requires an embedder).
* `Ekimetrics.DCC` — contextual coherence: chunk-vs-context-window embedding
  cosine (requires an embedder).

Contract constraints, load-bearing:

* **FMRE/RC excluded.** The coref-dependent "Filtered Missing Reference Error"
  is not lifted: it requires `maverick-coref` (CC BY-NC-SA 4.0) and its
  upstream RC=99.0 figure is unverified (reference-boundary corruption fixed
  only on 2026-07-06). No coref code or license exposure enters this repo.
* **Never a gate.** Intrinsic scores are informational next to NID/TEDS/MHS and
  do not gate on their own; the Ekimetrics-vs-HOPE side-by-side is the decision
  instrument (intake-581 falsifies the cohesion premise ICC/BI rest on).
* **Embedder degrade.** ICC/DCC require a sentence-transformers model; when
  none is provided the rows carry `value=None` with the reason in `detail`,
  exactly like a missing extraction backend reports `available=False`. The
  default token counter is a deterministic whitespace approximation of
  upstream's tiktoken counter (pass `count_tokens_func` for exact numbers).

Implementation: `intrinsic.py` (MIT-attributed lift of the four non-coref
metrics + `DefaultChunker`, a deterministic heading/paragraph-aware splitter
mirroring the Phase-2 chunker direction; the real Phase-2 chunker slots in by
passing its chunks to `score_chunks` directly).

## Venv topology (three interpreters — this is load-bearing)

| interpreter | pdftotext | ODL-local | LiteParse | bench scoring |
|---|---|---|---|---|
| research `.venv` (py3.14) | ✅ | ❌ no `opendataloader_pdf` | ❌ | ❌ no `Levenshtein/apted` |
| orchestrator `.venv` | ✅ | ✅ has `opendataloader_pdf` | ❌ | ❌ |
| bench `.venv` (py3.11) | ✅ | ❌ | ❌ | ✅ |

Consequences, all handled by the adapter (backends never crash on absence — they
return `available=False`):

* **Generate predictions** with pdftotext from *any* interpreter; with **ODL-local
  you must use the orchestrator venv** (or `pip install opendataloader-pdf` into the
  research venv). Run `… adapter availability` to see what's live where.
* **Scoring** always runs as a subprocess pinned to the **bench venv**
  (`bootstrap.bench_python()`), never in-process — the research venv can't import
  the bench (py3.14 vs the bench's `>=3.10,<3.12`, and no scorer deps).

`liteparse` is **not installed on any EPYC venv** at wiring time, and it is **not a
`pdf_router` backend** (it lives only in the orchestrator's `pdf_fastpath_probe.py`).
Our `LiteParseBackend` mirrors that probe's invocation contract and reports
`available=False` until the module is installed.

## Input-modality gap (precondition for real scored rows)

`demo_data/` ships **page images (`.jpg`) + reference markdown, NOT source PDFs.**
Our deterministic backends need **born-digital PDFs with a text layer**. Therefore a
real GT-scored comparison requires a `pdf_manifest` mapping each GT page's
`image_path` basename → a source PDF:

```json
{ "yanbaopptmerge_SE05.pdf_7.jpg": "/path/to/source.pdf", "...": "..." }
```

or `{"pairs": [{"gt_image": "...", "pdf": "..."}]}`. GT pages with no mapped PDF are
skipped deterministically. Sourcing the OmniDocBench PDFs (Git-LFS dataset absent
locally) — or supplying a local born-digital corpus with its own GT — is the
standing precondition captured in the Wave-3 stubs.

## Wave-3 PaddleOCR-VL Producer

`run-model --engine paddleocr_vl_1_6` is the first runnable model-gated producer.
It consumes GT page images directly instead of a PDF manifest, launches a guarded
experimental-v7 PaddleOCR-VL `llama-server`, writes one Markdown prediction per
page, records per-page response JSON, then optionally scores the prediction dir
with the same OmniDocBench config. It requires `--allow-inference`; without that
flag argparse exits before any server launch. Per-page model failures become
empty prediction artifacts plus error JSON so one malformed server response does
not discard the rest of a scored run.

First operational demo: `/mnt/raid0/llm/tmp/odl-paddleocr-vl-demo-20260717T200212Z/`.
It wrote and scored all `18` demo predictions, captured one `peg-native` model
error as an empty page, and reported median decode `485.30 t/s`, median page
latency `2918.78 ms`, text-block edit distance `0.343019`, reading-order edit
distance `0.337318`, and table TEDS `0.0`. Treat that as producer/runtime
evidence plus a table-format prompt gap, not a final document-parser quality
claim.

Follow-up `/mnt/raid0/llm/tmp/odl-paddleocr-vl-htmltables-20260717T201106Z/`
used `--prompt-profile html_tables`. It completed without model errors and
improved reading-order edit distance to `0.285753`, but emitted zero HTML
`<table>` tags, kept table TEDS at `0.0`, worsened text-block edit distance to
`0.429062`, and slowed median page latency to `3245.60 ms`. Prompt-only table
recovery is therefore negative; the next table lever should be post-processing /
HTML conversion or a different parser, not another near-identical prompt.

## Usage

```bash
RES=/mnt/raid0/llm/epyc-inference-research
# what's available in THIS interpreter
$RES/.venv/bin/python -m scripts.benchmark.odl_bench.adapter availability

# generate predictions (+score) for all deterministic engines — use the
# ORCHESTRATOR venv so ODL-local is live; scoring auto-subprocesses to the bench venv
/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python \
  -m scripts.benchmark.odl_bench.adapter run \
  --gt   /mnt/raid0/llm/opendataloader-bench/demo_data/omnidocbench_demo/OmniDocBench_demo.json \
  --pdf-manifest <pdf_manifest.json> --run-dir /tmp/odlrun --score

# emit the Wave-3 model-gated manifest stubs (JSON)
$RES/.venv/bin/python -m scripts.benchmark.odl_bench.adapter stubs

# run the PaddleOCR-VL document-parser arm over GT page images, then optionally score
$RES/.venv/bin/python -m scripts.benchmark.odl_bench.adapter run-model \
  --engine paddleocr_vl_1_6 \
  --gt /mnt/raid0/llm/opendataloader-bench/demo_data/omnidocbench_demo/OmniDocBench_demo.json \
  --image-root /mnt/raid0/llm/opendataloader-bench/demo_data/omnidocbench_demo/images \
  --run-dir /mnt/raid0/llm/tmp/odl-paddleocr-vl \
  --prompt-profile html_tables \
  --allow-inference --score
```

Library API: `OdlBenchAdapter.{generate_predictions, emit_config, score,
build_deterministic_row_set, build_model_gated_row_set, parse_metric_result,
score_intrinsic, model_gated_manifest_stubs}` plus `intrinsic.{score_chunks,
score_prediction_dir, DefaultChunker}`.

## B2 import contract (read-only symbols we depend on)

Preserved by the sibling B2 agent in `epyc-orchestrator/src/services/pdf_router.py`:

```python
from src.services.pdf_router import PDFRouter, PDFExtractionResult
PDFRouter()                                              # instantiable with defaults
PDFRouter._extract_with_pdftotext(self, Path) -> (text: str, latency_ms: float)
PDFRouter._extract_with_opendataloader(self, Path) -> (markdown: str, latency_ms: float)
```

If B2 renames/moves these, **only `backends.py` changes.** The resolver
(`backends.get_pdf_router`) bootstraps the orchestrator onto `sys.path`
(`$EPYC_ORCHESTRATOR_ROOT` → `/workspace/repos/…` → `/mnt/raid0/llm/…`) and caches
one `PDFRouter` instance; on any import/instantiation failure it degrades to
`available=False` with the reason recorded, so a sweep never blocks on B2.

## Tests

Stdlib `unittest` (research repo has no pytest), deterministic, no inference:

```bash
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
  scripts/benchmark/odl_bench/tests/test_odl_bench.py
/mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
  scripts/benchmark/odl_bench/tests/test_intrinsic.py
```

`test_odl_bench.py` covers: naming contract vs real `demo_data` fixtures,
fake-backend prediction generation + speed rows, **real pdftotext on an
in-test generated born-digital PDF**, config emission/loadability, metric-result
parsing (known nesting + missing-key robustness), model-gated stub
completeness/exclusion, engine availability report, and the scoring-command
interpreter pin. `test_intrinsic.py` covers the Ekimetrics metrics (SC/BI math
with known fixtures, ICC/DCC against a deterministic fake embedder, FMRE
exclusion, no-embedder degrade, default chunker determinism, per-dir
aggregation).

## Files

| file | role |
|---|---|
| `bootstrap.py` | locate orchestrator + bench roots; put orchestrator on `sys.path`; bench interpreter |
| `backends.py` | deterministic backend resolver (pdftotext/ODL-local/LiteParse) + FakeBackend + registry |
| `schemas.py` | row/manifest/stub dataclasses |
| `run_configs.py` | OmniDocBench config template + metric mapping + naming contract |
| `adapter.py` | `OdlBenchAdapter` (generate → config → score → rows) + CLI |
| `intrinsic.py` | Ekimetrics intrinsic chunk-quality metrics (SC/BI/ICC/DCC; FMRE excluded) + default chunker |
| `manifest_stubs.py` | model-gated Wave-3 manifest-entry stubs |
| `paddleocr_vl.py` | guarded PaddleOCR-VL image→markdown producer |
| `tests/test_odl_bench.py` | stdlib-runnable deterministic tests |
| `tests/test_intrinsic.py` | stdlib-runnable Ekimetrics intrinsic metric tests |
