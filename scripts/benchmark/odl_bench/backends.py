"""Deterministic extraction backends, resolved from the orchestrator pdf_router.

The three deterministic engines wired into the bench:

  * ``pdftotext``      -> ``PDFRouter._extract_with_pdftotext``   (poppler; always available)
  * ``opendataloader`` -> ``PDFRouter._extract_with_opendataloader`` (ODL-local; needs the
                          ``opendataloader_pdf`` module + Java 11+; rule-based / deterministic)
  * ``liteparse``      -> dynamic ``liteparse`` module (optional; JVM-free born-digital
                          fast-path; NOT a pdf_router backend — mirrors the contract in
                          ``epyc-orchestrator/scripts/benchmark/pdf_fastpath_probe.py``)

B2 IMPORT CONTRACT (read-only; symbols we depend on, preserved by the B2 agent):
    from src.services.pdf_router import PDFRouter, PDFExtractionResult
    PDFRouter()                                  # instantiable with defaults
    PDFRouter._extract_with_pdftotext(self, Path) -> tuple[str, float]   # (text, latency_ms)
    PDFRouter._extract_with_opendataloader(self, Path) -> tuple[str, float]
If any of these move/rename, only THIS file changes.

Every backend answers ``available()`` truthfully for the CURRENT interpreter and
NEVER raises on run — unavailability degrades to an ``ExtractionOutcome`` with
``available=False`` so a bench sweep records "engine absent here" rather than
crashing. This is what lets the same adapter run under the research venv
(pdftotext only) or the orchestrator venv (pdftotext + ODL-local).
"""

from __future__ import annotations

import importlib
import importlib.util
import time
from pathlib import Path
from typing import Callable

from . import bootstrap
from .schemas import ExtractionOutcome

DETERMINISTIC_ENGINES = ("pdftotext", "opendataloader", "liteparse")

# ---------------------------------------------------------------------------
# Orchestrator PDFRouter resolver (cached)
# ---------------------------------------------------------------------------
_ROUTER = None
_ROUTER_ERROR = ""


def get_pdf_router():
    """Return a cached orchestrator ``PDFRouter`` instance, or None if unavailable.

    Best-effort: bootstraps the orchestrator onto sys.path, imports the class, and
    instantiates it with defaults. Failure is recorded (``router_error()``) and
    None is returned so callers fall back to fake/absent backends.
    """
    global _ROUTER, _ROUTER_ERROR
    if _ROUTER is not None:
        return _ROUTER
    if _ROUTER_ERROR:
        return None

    root = bootstrap.ensure_orchestrator_on_path()
    if root is None:
        _ROUTER_ERROR = "epyc-orchestrator checkout not found on any known path"
        return None
    try:
        from src.services.pdf_router import PDFRouter  # type: ignore
    except Exception as exc:  # pragma: no cover - env dependent
        _ROUTER_ERROR = f"import PDFRouter failed: {type(exc).__name__}: {exc}"
        return None
    try:
        _ROUTER = PDFRouter()
    except Exception as exc:  # pragma: no cover - env dependent
        _ROUTER_ERROR = f"PDFRouter() instantiation failed: {type(exc).__name__}: {exc}"
        return None
    return _ROUTER


def router_error() -> str:
    return _ROUTER_ERROR


def reset_router_cache() -> None:
    """Testing hook: drop the cached router so a fresh resolve happens."""
    global _ROUTER, _ROUTER_ERROR
    _ROUTER = None
    _ROUTER_ERROR = ""


# ---------------------------------------------------------------------------
# Backend base + concrete deterministic backends
# ---------------------------------------------------------------------------
class Backend:
    name: str = ""
    kind: str = "deterministic"

    def available(self) -> tuple[bool, str]:
        """(is_available_in_this_interpreter, reason_if_not)."""
        raise NotImplementedError

    def run(self, pdf_path: Path) -> ExtractionOutcome:
        raise NotImplementedError

    # shared helper
    def _unavailable(self, pdf_path: Path, detail: str) -> ExtractionOutcome:
        return ExtractionOutcome(
            engine=self.name,
            pdf_path=str(pdf_path),
            available=False,
            ok=False,
            text="",
            latency_ms=0.0,
            method=self.name,
            detail=detail,
        )


class PdftotextBackend(Backend):
    name = "pdftotext"

    def available(self) -> tuple[bool, str]:
        router = get_pdf_router()
        if router is None:
            return False, router_error() or "PDFRouter unavailable"
        import shutil

        path = getattr(router, "pdftotext_path", "pdftotext")
        if shutil.which(path) is None and not Path(path).exists():
            return False, f"pdftotext binary not found at {path!r}"
        return True, ""

    def run(self, pdf_path: Path) -> ExtractionOutcome:
        ok_avail, reason = self.available()
        if not ok_avail:
            return self._unavailable(pdf_path, reason)
        router = get_pdf_router()
        text, latency_ms = router._extract_with_pdftotext(Path(pdf_path))
        return ExtractionOutcome(
            engine=self.name,
            pdf_path=str(pdf_path),
            available=True,
            ok=bool(text.strip()),
            text=text,
            latency_ms=float(latency_ms),
            method="pdftotext",
            char_count=len(text),
        )


class OpendataloaderLocalBackend(Backend):
    """ODL-local (rule-based XY-Cut++; deterministic; markdown output)."""

    name = "opendataloader"

    def available(self) -> tuple[bool, str]:
        router = get_pdf_router()
        if router is None:
            return False, router_error() or "PDFRouter unavailable"
        if importlib.util.find_spec("opendataloader_pdf") is None:
            return (
                False,
                "opendataloader_pdf not importable in this interpreter "
                "(present in the orchestrator venv, absent in the research venv) — "
                "run this adapter under a python that has it, or pip install it",
            )
        return True, ""

    def run(self, pdf_path: Path) -> ExtractionOutcome:
        ok_avail, reason = self.available()
        if not ok_avail:
            return self._unavailable(pdf_path, reason)
        router = get_pdf_router()
        text, latency_ms = router._extract_with_opendataloader(Path(pdf_path))
        return ExtractionOutcome(
            engine=self.name,
            pdf_path=str(pdf_path),
            available=True,
            ok=bool(text.strip()),
            text=text,
            latency_ms=float(latency_ms),
            method="opendataloader_local",
            char_count=len(text),
            detail="" if text.strip() else "ODL returned empty output",
        )


class LiteParseBackend(Backend):
    """Optional JVM-free born-digital fast-path (run-llama/liteparse).

    Not a pdf_router backend. Mirrors the invocation contract used by the
    orchestrator's ``pdf_fastpath_probe.py`` so results are comparable. Absent
    from every EPYC venv at wiring time -> reports unavailable, never errors.
    """

    name = "liteparse"

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("liteparse") is None:
            return False, "liteparse module not installed on any EPYC venv"
        return True, ""

    def _instantiate(self, module):
        for cls_name in ("LiteParse", "Parser", "DocumentParser"):
            cls = getattr(module, cls_name, None)
            if cls is None:
                continue
            for kwargs in ({"ocr_enabled": False}, {"ocr": False}, {}):
                try:
                    return cls(**kwargs)
                except TypeError:
                    continue
        raise AttributeError("liteparse exposes no LiteParse/Parser/DocumentParser")

    @staticmethod
    def _coerce_text(result) -> str:
        for attr in ("markdown", "text", "content"):
            val = getattr(result, attr, None)
            if isinstance(val, str) and val.strip():
                return val
        if isinstance(result, str):
            return result
        return str(result) if result is not None else ""

    def run(self, pdf_path: Path) -> ExtractionOutcome:
        ok_avail, reason = self.available()
        if not ok_avail:
            return self._unavailable(pdf_path, reason)
        start = time.perf_counter()
        try:
            module = importlib.import_module("liteparse")
            if hasattr(module, "parse"):
                result = module.parse(str(pdf_path))
            else:
                parser = self._instantiate(module)
                for method_name in ("parse", "load", "extract", "convert"):
                    method = getattr(parser, method_name, None)
                    if method is not None:
                        result = method(str(pdf_path))
                        break
                else:
                    result = parser(str(pdf_path)) if callable(parser) else None
            text = self._coerce_text(result)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ExtractionOutcome(
                engine=self.name,
                pdf_path=str(pdf_path),
                available=True,
                ok=bool(text.strip()),
                text=text,
                latency_ms=latency_ms,
                method="liteparse",
                char_count=len(text),
            )
        except Exception as exc:  # pragma: no cover - only with liteparse installed
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ExtractionOutcome(
                engine=self.name,
                pdf_path=str(pdf_path),
                available=True,
                ok=False,
                text="",
                latency_ms=latency_ms,
                method="liteparse",
                detail=f"{type(exc).__name__}: {exc}",
            )


class FakeBackend(Backend):
    """Deterministic in-memory backend for tests (no orchestrator, no inference).

    ``render`` maps a PDF path -> markdown string; defaults to a stable stub.
    """

    def __init__(self, name: str = "fake", render: Callable[[Path], str] | None = None,
                 latency_ms: float = 1.0):
        self.name = name
        self._render = render or (lambda p: f"# {Path(p).stem}\n\nfake extraction\n")
        self._latency = latency_ms

    def available(self) -> tuple[bool, str]:
        return True, ""

    def run(self, pdf_path: Path) -> ExtractionOutcome:
        text = self._render(Path(pdf_path))
        return ExtractionOutcome(
            engine=self.name,
            pdf_path=str(pdf_path),
            available=True,
            ok=bool(text.strip()),
            text=text,
            latency_ms=self._latency,
            method=self.name,
            char_count=len(text),
        )


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    """Register/override a backend by name (tests inject FakeBackend here)."""
    _REGISTRY[backend.name] = backend


def unregister_backend(name: str) -> None:
    _REGISTRY.pop(name, None)


def resolve_backend(name: str) -> Backend:
    """Return a Backend by name, honouring test overrides first."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    if name == "pdftotext":
        return PdftotextBackend()
    if name == "opendataloader":
        return OpendataloaderLocalBackend()
    if name == "liteparse":
        return LiteParseBackend()
    raise ValueError(f"unknown backend {name!r}; known: {DETERMINISTIC_ENGINES}")


def availability_report() -> dict[str, dict[str, str]]:
    """Availability of every deterministic engine in the CURRENT interpreter."""
    report: dict[str, dict[str, str]] = {}
    for name in DETERMINISTIC_ENGINES:
        avail, reason = resolve_backend(name).available()
        report[name] = {"available": str(avail), "reason": reason}
    return report
