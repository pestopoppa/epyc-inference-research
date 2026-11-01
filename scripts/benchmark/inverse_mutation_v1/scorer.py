"""Pure-format exact-match scorer for CJ-7a; never imports or executes candidate code."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .fixtures import Fixture, get_fixture
from ..gate_verdict_vocab import (
    CAUSE_ABSENT,
    CAUSE_CHECKER_ERROR,
    CAUSE_NO_REFERENCE,
    CAUSE_TIMEOUT,
    CAUSE_UNSUPPORTED,
    VERDICT_FAIL,
    VERDICT_PASS,
    out_of_coverage,
    verdict_of,
)

RUFF_VERSION = "ruff 0.16.3"
RUFF_SHA256 = "88abf248be39383e05419e50811467b9c25ef3f3bc1b8a0874962941c0190ffa"
FORMAT_CONFIG = "isolated; line-length=100"
FORMAT_TIMEOUT_SECONDS = 3.0


class FormatUnknown(RuntimeError):
    def __init__(self, cause: str, detail: str, formatter: dict | None = None):
        super().__init__(detail)
        self.cause = cause
        self.formatter = formatter or {
            "version": None,
            "configuration": FORMAT_CONFIG,
            "executable_sha256": None,
        }


def _formatter() -> tuple[Path, dict[str, str | None]]:
    metadata: dict[str, str | None] = {
        "version": None,
        "configuration": FORMAT_CONFIG,
        "executable_sha256": None,
    }
    executable = shutil.which("ruff")
    if executable is None:
        raise FormatUnknown(CAUSE_UNSUPPORTED, "pinned Ruff formatter is unavailable", metadata)
    path = Path(executable).resolve()
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FormatUnknown(
            CAUSE_CHECKER_ERROR, "Ruff executable cannot be read", metadata
        ) from exc
    metadata["executable_sha256"] = digest
    if digest != RUFF_SHA256:
        raise FormatUnknown(
            CAUSE_UNSUPPORTED, "Ruff executable does not match pinned digest", metadata
        )
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=FORMAT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FormatUnknown(CAUSE_TIMEOUT, "Ruff version probe timed out", metadata) from exc
    except OSError as exc:
        raise FormatUnknown(CAUSE_CHECKER_ERROR, "Ruff version probe failed", metadata) from exc
    metadata["version"] = result.stdout.strip() or None
    if result.returncode != 0 or result.stdout.strip() != RUFF_VERSION:
        raise FormatUnknown(
            CAUSE_UNSUPPORTED, "Ruff version does not match pinned version", metadata
        )
    return path, metadata


def _format_with(source: str, executable: Path, formatter: dict[str, str | None]) -> str:
    try:
        result = subprocess.run(
            [
                str(executable),
                "format",
                "--isolated",
                "--line-length",
                "100",
                "--stdin-filename",
                "fixture.py",
                "-",
            ],
            input=source,
            capture_output=True,
            text=True,
            timeout=FORMAT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FormatUnknown(CAUSE_TIMEOUT, "Ruff formatting timed out", formatter) from exc
    except OSError as exc:
        raise FormatUnknown(
            CAUSE_CHECKER_ERROR, "Ruff formatter could not start", formatter
        ) from exc
    if result.returncode != 0:
        raise FormatUnknown(CAUSE_CHECKER_ERROR, "Ruff could not format candidate", formatter)
    return result.stdout


def _sha256_text(value: str | None) -> str | None:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value is not None else None


def score(fixture_id: str, candidate: str | None) -> dict:
    """Return the native score record, including contemporaneous pins and content hashes."""
    recorded_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    candidate_raw_sha256 = _sha256_text(candidate)
    base = {
        "recorded_at_utc": recorded_at,
        "fixture_id": fixture_id,
        "instruction": None,
        "source_sha256": None,
        "source_provenance": None,
        "mutation": None,
        "instruction_sha256": None,
        "candidate_raw_sha256": candidate_raw_sha256,
        "candidate_normalized_sha256": None,
        "expected_normalized_sha256": None,
        "formatter": {
            "version": None,
            "configuration": FORMAT_CONFIG,
            "executable_sha256": None,
        },
    }
    if candidate is None:
        return {
            **base,
            **out_of_coverage(fixture_id, CAUSE_ABSENT, "no candidate source was supplied"),
        }
    try:
        fixture: Fixture = get_fixture(fixture_id)
    except (KeyError, ValueError) as exc:
        return {**base, **out_of_coverage(fixture_id, CAUSE_NO_REFERENCE, str(exc))}
    base.update(
        {
            "source_sha256": fixture.provenance["source_sha256"],
            "source_provenance": fixture.provenance,
            "mutation": fixture.mutation,
            "instruction": fixture.instruction,
            "instruction_sha256": _sha256_text(fixture.instruction),
        }
    )
    try:
        executable, formatter = _formatter()
        base["formatter"] = formatter
        expected = _format_with(fixture.expected_source, executable, formatter)
        actual = _format_with(candidate, executable, formatter)
    except FormatUnknown as exc:
        base["formatter"] = exc.formatter
        return {**base, **out_of_coverage(fixture_id, exc.cause, str(exc))}
    base["candidate_normalized_sha256"] = _sha256_text(actual)
    base["expected_normalized_sha256"] = _sha256_text(expected)
    verdict = VERDICT_PASS if actual == expected else VERDICT_FAIL
    return {**base, **verdict_of(fixture_id, verdict)}
