#!/usr/bin/env python3
"""ODL-013 three-way bench: LiteParse-local vs OpenDataLoader-local vs pdftotext.

Benchmarks the three deterministic born-digital PDF backends against the
opendataloader-bench 200-PDF corpus (ground-truth markdown), scored with the
upstream harness's own evaluator (NID = reading order, TEDS = table fidelity,
MHS = heading fidelity, plus per-doc latency = speed).

Fail-closed contract (PIP-05 hardening, 2026-08-25):
  * a non-zero extractor return code, a missing candidate output, or an empty
    prediction FAILS the document — the prediction file is never left behind
    as if it were a valid prediction;
  * any failed document makes the parse phase exit non-zero (after writing the
    per-engine summaries so the failures are forensically visible);
  * the ODL/LiteParse engine versions are pinned (ENGINE_PINS); the parse
    phase refuses to run when the installed distribution is missing or does
    not match the pin;
  * raw per-document latencies are persisted (summary["per_doc_latency_ms"]) so
    the reported median/p90 can be reproduced independently.

Layout (mirrors the upstream harness's prediction/engine/markdown contract):

    <run_dir>/prediction/<engine>/markdown/<stem>.md
    <run_dir>/prediction/<engine>/summary.json     (engine + timing metadata)

Two-phase split because no single EPYC venv has both engine deps and scorer
deps:

    Phase 1 (PARSE) — research venv, python3.13:
        liteparse 2.12.0 (cp313 wheel) + opendataloader-pdf 2.5.0 (py3-none-any)
        + pdftotext (system poppler binary).
    Phase 2 (SCORE) — the OmniDocBench clone's .venv (omnidocbench), python3.11:
        apted, rapidfuzz, lxml, bs4 (the upstream evaluator's deps).

Phase 1 emits predictions; Phase 2 imports the upstream evaluator read-only and
scores each engine dir. The upstream clone at /mnt/raid0/llm/opendataloader-bench-upstream
is NEVER modified (read-only import + read-only ground-truth).

Usage:
    # Phase 1 — parse all engines (needs liteparse + opendataloader_pdf importable)
    $RES/.venv/bin/python -m scripts.benchmark.odl_bench.run_three_way_bench parse \
        --corpus /mnt/raid0/llm/opendataloader-bench-upstream --run-dir <dir>

    # Phase 2 — score (needs apted/rapidfuzz/lxml/bs4 importable)
    $BENCH/.venv/bin/python -m scripts.benchmark.odl_bench.run_three_way_bench score \
        --corpus /mnt/raid0/llm/opendataloader-bench-upstream --run-dir <dir> \
        --report <run_dir>/three_way_report.md

LiteParse-output-awareness: liteparse is invoked with output_format="markdown"
(the official SDK option) so its tables/headings are markdown-parseable by the
upstream TEDS/MHS evaluators. The raw non-markdown layout output (which breaks
naive TEDS) is intentionally NOT the scored artifact — see intake-646 Tier 2b.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ENGINES = ("pdftotext", "opendataloader", "liteparse")

LITEPARSE_OUTPUT_FORMAT = "markdown"  # LiteParse-output-aware mode (SDK option)

# PIP-05: engine runs must be attributable to a pinned version. The parse
# phase fails closed when the installed distribution cannot be resolved or
# does not match the pin (opendataloader-pdf 2.5.0 and liteparse 2.12.0 are
# the versions pinned for the 2026-08-13 ODL-013 run).
ENGINE_PINS = {
    "opendataloader": "opendataloader-pdf==2.5.0",
    "liteparse": "liteparse==2.12.0",
}


def _installed_dist_version(dist_name: str) -> str | None:
    """Installed distribution version via importlib.metadata, or None.

    None means the version cannot be attributed — treated as fail-closed.
    """
    try:
        from importlib import metadata

        return metadata.version(dist_name)
    except Exception:  # noqa: BLE001 - any resolution failure means unpinned
        return None


def _resolve(module: str):
    """Import helper with a friendly failure for the split-venv contract."""
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as exc:
        sys.exit(
            f"Module {module!r} not importable in this interpreter "
            f"({sys.executable}). Phase 1 (parse) must run under the research "
            f"venv (liteparse + opendataloader_pdf); Phase 2 (score) must run "
            f"under the omnidocbench .venv (apted/rapidfuzz/lxml/bs4). "
            f"Detail: {exc}"
        )


# ---------------------------------------------------------------------------
# Phase 1 — parse
# ---------------------------------------------------------------------------
def _parse_pdftotext(pdf: Path, out_dir: Path, stem: str) -> float:
    """pdftotext -layout -> <stem>.md (plain text; reading-order faithful).

    Fail closed: a non-zero exit code raises instead of returning a partial
    or empty prediction.
    """
    start = time.perf_counter()
    r = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    latency = (time.perf_counter() - start) * 1000.0
    if r.returncode != 0:
        detail = (r.stderr or r.stdout or "").strip()[:500] or "no detail"
        raise RuntimeError(f"pdftotext exited {r.returncode}: {detail}")
    text = r.stdout
    (out_dir / f"{stem}.md").write_text(text, encoding="utf-8")
    return latency


def _parse_opendataloader(pdf: Path, out_dir: Path, stem: str) -> float:
    """opendataloader_pdf.convert (local, rule-based) -> <stem>.md.

    Fail closed: a missing candidate output raises instead of silently
    writing an empty prediction.
    """
    import opendataloader_pdf

    start = time.perf_counter()
    opendataloader_pdf.convert(
        input_path=[str(pdf)],
        output_dir=str(out_dir),
        format=["markdown"],
        table_method="cluster",
        image_output="off",
        quiet=True,
    )
    latency = (time.perf_counter() - start) * 1000.0
    md_file = out_dir / f"{stem}.md"
    if not md_file.exists():
        # SDK may write a sibling beside source or with different casing; scan
        # the output dir for the stem before failing closed.
        candidates = [p for p in out_dir.glob(f"{stem}*") if p.suffix in (".md", ".MD")]
        if not candidates:
            raise RuntimeError(
                f"opendataloader produced no markdown candidate for {stem}"
            )
        md_file = candidates[0]
    return latency


def _parse_liteparse(pdf: Path, out_dir: Path, stem: str) -> float:
    """liteparse 2.12 with output_format=markdown (LiteParse-output-aware).

    Fail closed: an empty prediction raises instead of being scored as
    success.
    """
    from liteparse import LiteParse

    start = time.perf_counter()
    lp = LiteParse(output_format=LITEPARSE_OUTPUT_FORMAT)
    result = lp.parse(str(pdf))
    latency = (time.perf_counter() - start) * 1000.0
    if not result.text.strip():
        raise RuntimeError(f"liteparse returned an empty prediction for {stem}")
    (out_dir / f"{stem}.md").write_text(result.text, encoding="utf-8")
    return latency


def _remove_prediction(out_dir: Path, stem: str) -> None:
    """Never leave a failed doc's (possibly partial) prediction on disk."""
    for stale in out_dir.glob(f"{stem}*"):
        stale.unlink()


_PARSERS = {
    "pdftotext": _parse_pdftotext,
    "opendataloader": _parse_opendataloader,
    "liteparse": _parse_liteparse,
}


def phase_parse(corpus: Path, run_dir: Path, engines: tuple[str, ...]) -> None:
    pdf_dir = corpus / "pdfs"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {pdf_dir}")

    import cpuinfo  # summary metadata parity with upstream

    any_failed = False
    for engine in engines:
        out_dir = run_dir / "prediction" / engine / "markdown"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Wipe stale predictions from a previous run of the same engine.
        for stale in out_dir.glob("*.md"):
            stale.unlink()

        engine_version = _pinned_engine_version(engine)
        parser = _PARSERS[engine]
        per_doc_latency: dict[str, float] = {}
        failed_docs: list[str] = []
        started = time.time()
        for pdf in pdfs:
            stem = pdf.stem
            try:
                lat = parser(pdf, out_dir, stem)
            except Exception as exc:  # noqa: BLE001 — record per-doc failure
                failed_docs.append(stem)
                _remove_prediction(out_dir, stem)
                print(f"  [{engine}] {stem}: ERROR {type(exc).__name__}: {exc}")
                continue
            pred_file = out_dir / f"{stem}.md"
            if not pred_file.exists() or not pred_file.read_text(
                encoding="utf-8"
            ).strip():
                # Fail closed: an empty prediction must never look like success.
                failed_docs.append(stem)
                _remove_prediction(out_dir, stem)
                print(f"  [{engine}] {stem}: ERROR empty prediction")
                continue
            per_doc_latency[stem] = lat
        total = time.time() - started

        failed = len(failed_docs)
        if failed:
            any_failed = True
        latencies = list(per_doc_latency.values())
        summary = {
            "engine_name": engine,
            "engine_version": engine_version,
            "engine_pin": ENGINE_PINS.get(engine) or "system binary (unpinned)",
            "processor": cpuinfo.get_cpu_info()["brand_raw"],
            "document_count": len(pdfs),
            "total_elapsed": total,
            "elapsed_per_doc": (total / len(pdfs)) if pdfs else 0.0,
            "median_latency_ms": _median(latencies),
            "p50_latency_ms": _percentile(latencies, 50),
            "p90_latency_ms": _percentile(latencies, 90),
            "failed_docs": failed,
            "failed_stems": failed_docs,
            "latency_count": len(latencies),
            "per_doc_latency_ms": {
                stem: round(ms, 6) for stem, ms in sorted(per_doc_latency.items())
            },
            "date": time.strftime("%Y-%m-%d"),
        }
        (run_dir / "prediction" / engine / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"[parse] {engine}: {len(pdfs)} docs, {total:.1f}s total, "
              f"{summary['median_latency_ms']:.1f} ms median, {failed} failed")
    if any_failed:
        # Fail closed: a run with failed documents must not be consumable as a
        # clean run. Summaries were written above so failures are inspectable.
        print(
            "[parse] FAILED CLOSED: one or more engines had failed documents; "
            "predictions for failed docs were NOT written (see per-engine "
            "summary.json 'failed_stems')",
            file=sys.stderr,
        )
        sys.exit(2)


def _pinned_engine_version(engine: str) -> str:
    """Resolve the engine version against ENGINE_PINS; fail closed otherwise.

    Unpinned engines (pdftotext system binary) keep the plain version probe.
    """
    pin = ENGINE_PINS.get(engine)
    if pin is None:
        return _engine_version(engine)
    dist, _, want = pin.partition("==")
    installed = _installed_dist_version(dist)
    if installed is None:
        print(
            f"[{engine}] installed distribution {dist!r} version unresolvable; "
            f"cannot attribute the run to pinned {pin} — fail closed",
            file=sys.stderr,
        )
        sys.exit(2)
    if installed != want:
        print(
            f"[{engine}] installed {dist}=={installed} != pinned {pin} — "
            "fail closed (run would not be attributable to the pinned profile)",
            file=sys.stderr,
        )
        sys.exit(2)
    return installed


def _engine_version(engine: str) -> str:
    """Version probe for unpinned (system-binary) engines, e.g. pdftotext."""
    try:
        if engine == "pdftotext":
            r = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True)
            m = re.search(r"version (\S+)", r.stderr or r.stdout)
            return m.group(1) if m else "?"
    except Exception:  # noqa: BLE001
        pass
    return "?"


def _median(values: list[float]) -> float:
    return _percentile(values, 50)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# Phase 2 — score with the upstream evaluator (read-only)
# ---------------------------------------------------------------------------
def phase_score(corpus: Path, run_dir: Path, engines: tuple[str, ...],
                report_path: Path | None = None) -> dict[str, dict]:
    """Score each engine dir via the upstream evaluator; returns engine->metrics.

    Fail closed: a missing prediction dir, a missing or empty prediction for
    any ground-truth document, or a missing evaluation.json aborts with a
    non-zero exit code instead of silently skipping the engine.
    """

    # Read-only import of the upstream evaluator (never modified).
    src = corpus / "src"
    if not src.is_dir():
        sys.exit(f"Upstream src dir not found at {src}")
    sys.path.insert(0, str(src))
    from evaluator import _evaluate_engine_version  # type: ignore

    gt_dir = corpus / "ground-truth" / "markdown"
    if not gt_dir.is_dir():
        sys.exit(f"Ground-truth markdown dir not found at {gt_dir}")
    gt_stems = sorted(p.stem for p in gt_dir.glob("*.md"))

    results: dict[str, dict] = {}
    for engine in engines:
        pred_dir = run_dir / "prediction" / engine
        pred_md = pred_dir / "markdown"
        if not pred_md.is_dir():
            print(f"[score] {engine}: prediction markdown dir missing — fail closed",
                  file=sys.stderr)
            sys.exit(3)
        missing = [s for s in gt_stems if not (pred_md / f"{s}.md").is_file()]
        empty = [
            s for s in gt_stems
            if (pred_md / f"{s}.md").is_file()
            and not (pred_md / f"{s}.md").read_text(encoding="utf-8").strip()
        ]
        if missing or empty:
            print(
                f"[score] {engine}: FAILED CLOSED — missing predictions: "
                f"{missing[:10]} (n={len(missing)}); empty predictions: "
                f"{empty[:10]} (n={len(empty)})",
                file=sys.stderr,
            )
            sys.exit(3)
        _evaluate_engine_version(gt_dir, pred_dir, "evaluation.json")
        eval_path = pred_dir / "evaluation.json"
        if not eval_path.exists():
            print(f"[score] {engine}: evaluation.json not produced — fail closed",
                  file=sys.stderr)
            sys.exit(3)
        with eval_path.open(encoding="utf-8") as fh:
            ev = json.load(fh)
        metrics = ev.get("metrics", {}).get("score", {})
        summary = ev.get("summary", {})
        results[engine] = {
            "nid": metrics.get("nid_mean"),
            "teds": metrics.get("teds_mean"),
            "mhs": metrics.get("mhs_mean"),
            "overall": metrics.get("overall_mean"),
            "elapsed_per_doc": summary.get("elapsed_per_doc"),
            "engine_version": summary.get("engine_version"),
        }
        print(f"[score] {engine}: NID {results[engine]['nid']:.4f} | "
              f"TEDS {results[engine]['teds']:.4f} | "
              f"MHS {results[engine]['mhs']:.4f} | "
              f"overall {results[engine]['overall']:.4f}")

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_render_report(results, run_dir), encoding="utf-8")
        print(f"[score] report -> {report_path}")
    return results


def _render_report(results: dict[str, dict], run_dir: Path) -> str:
    lines = [
        "# ODL-013 — LiteParse-local vs OpenDataLoader-local vs pdftotext",
        "",
        f"Run dir: `{run_dir}`",
        f"Date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Scores (higher = better; NID/TEDS/MHS from upstream evaluator)",
        "",
        "| engine | NID (reading order) | TEDS (tables) | MHS (headings) | overall |",
        "|---|---|---|---|---|",
    ]
    for engine in ENGINES:
        r = results.get(engine)
        if not r:
            lines.append(f"| {engine} | — | — | — | — |")
            continue
        lines.append(
            f"| {engine} | {_fmt(r['nid'])} | {_fmt(r['teds'])} | "
            f"{_fmt(r['mhs'])} | {_fmt(r['overall'])} |"
        )
    lines += ["", "## Speed (per doc)", "",
              "| engine | median ms | p90 ms | elapsed_per_doc (s) |",
              "|---|---|---|---|"]
    for engine in ENGINES:
        summary_path = run_dir / "prediction" / engine / "summary.json"
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as fh:
                s = json.load(fh)
            lines.append(
                f"| {engine} | {s.get('median_latency_ms', 0.0):.1f} | "
                f"{s.get('p90_latency_ms', 0.0):.1f} | "
                f"{s.get('elapsed_per_doc', 0.0):.3f} |"
            )
        else:
            lines.append(f"| {engine} | — | — | — |")
    lines += ["", "## JVM-free deploy footprint", "",
              "- **pdftotext**: poppler binary; no JVM, no runtime deps.",
              "- **liteparse**: self-contained manylinux wheel (PDFium+tesseract compiled in); **no JVM**.",
              "- **opendataloader-local**: Python SDK wrapping a Java JAR; **requires Java 11+** (JVM spawn per convert())."]
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.4f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ("parse", "score"):
        p = sub.add_parser(cmd)
        p.add_argument("--corpus", type=Path,
                       default=Path("/mnt/raid0/llm/opendataloader-bench-upstream"))
        p.add_argument("--run-dir", type=Path, required=True)
        p.add_argument("--engines", default=",".join(ENGINES))
        if cmd == "score":
            p.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    engines = tuple(e.strip() for e in args.engines.split(",") if e.strip())
    unknown = set(engines) - set(ENGINES)
    if unknown:
        sys.exit(f"Unknown engine(s): {sorted(unknown)}; known: {ENGINES}")

    if args.cmd == "parse":
        phase_parse(args.corpus, args.run_dir, engines)
    else:
        phase_score(args.corpus, args.run_dir, engines, args.report)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
