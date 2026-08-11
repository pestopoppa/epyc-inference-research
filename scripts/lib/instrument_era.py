"""Instrument-era derivation. No era id is ever hard-coded by a caller.

Replaces the pinned-constant pattern that produced the 2026-07-29 kernel-era
mis-stamp: `ERA_CPU_KERNEL = "E6-cpu-kernel"` was correct when written on
2026-07-23 and silently wrong from the v8 cutover two days later. A constant
naming the *current* era goes stale at every cutover by construction; v9 proved
it a second time on 2026-08-11. Ordinals are not a fallback either — the
`cpu_bench` timeline is E0, E1, E5-cpu-kernel (v6+iqk), E6-cpu-kernel (**v7**),
E8-cpu-kernel (v8), E9-cpu-kernel (v9), with no E7-cpu-kernel at all.

Two derivations, deliberately different:

* :func:`derive_era` — "the era of scope S at instant T". Positional/date rule,
  used for scopes where every row is the same KIND of boundary.

* :func:`derive_kernel_era` — "which KERNEL was production at instant T". This
  may NOT use scope alone. Since the operator-signed consolidated token of
  2026-08-11T21:35Z, `cpu_bench` carries two kinds of boundary: kernel cutovers
  and `E8-cpu-bench-throttle-scope`, an ELIGIBILITY correction. A latest-in-scope
  lookup returns the eligibility row for any instant in
  2026-07-29..2026-08-10 — which is exactly the W1/W2/W4 window, so it would give
  the six known mis-stamped run manifests a second wrong answer. The
  discriminator is `binary_version`, added by
  RATIFY-CPU-BENCH-BINARY-VERSION-20260811: a row that carries one is a kernel
  cutover, a row that does not is not one.

* :func:`era_for_binary` — the strongest binding available. `binary_version` is
  the only field that WITNESSES which kernel executed a run, so a run manifest
  should resolve its era from the attested binary rather than from a date or,
  worse, by copying the cell template's stamp.

Everything fails CLOSED. `E5-cpu-kernel` deliberately carries no
`binary_version` — its registry note records none, and inventing one is the
precise failure this module exists to prevent — so kernel derivation raises for
instants in ``[2026-06-26T22:07:11Z, 2026-07-20T13:30:13Z)`` rather than
silently picking a neighbouring era. No banked E5 manifest falls in that window
(pre-registration begins 2026-07-23); the gap is real and is meant to be loud.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

INSTRUMENT_ERAS_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/orchestration/instrument_eras.yaml"
)
ERA_SOURCE = "epyc-orchestrator/orchestration/instrument_eras.yaml"

SCOPE_CPU_KERNEL = "cpu_bench"
SCOPE_EVAL_INSTRUMENT = "eval_quality"


class EraDerivationError(Exception):
    """Registry unreadable/ambiguous, or nothing witnesses the instant. Fatal."""


def _instant(value: Any, *, field: str, era_id: str = "?") -> _dt.datetime:
    """Normalise a registry/manifest instant to tz-aware UTC.

    The registry mixes full ISO timestamps with bare dates (``E1`` is
    ``2026-04-26``), and PyYAML hands those back as ``date``/``datetime``
    objects rather than strings, so a naive ``fromisoformat`` on everything
    raises. Naive values are read as UTC — the registry is documented UTC and
    every full-precision row carries ``Z``.
    """
    if isinstance(value, _dt.datetime):
        parsed = value
    elif isinstance(value, _dt.date):
        parsed = _dt.datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise EraDerivationError(f"era {era_id!r}: {field} is empty")
        try:
            parsed = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EraDerivationError(
                f"era {era_id!r}: {field}={value!r} is not ISO8601 ({exc})"
            ) from exc
    else:
        raise EraDerivationError(
            f"era {era_id!r}: {field}={value!r} has unsupported type {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Read + parse the era registry. Fails closed on every failure mode."""
    target = path or INSTRUMENT_ERAS_PATH
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment defect
        raise EraDerivationError(f"PyYAML unavailable: {exc}") from exc
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise EraDerivationError(f"cannot read era registry {target}: {exc}") from exc
    try:
        document = yaml.safe_load(raw)
    except Exception as exc:
        raise EraDerivationError(f"era registry {target} is not valid YAML: {exc}") from exc
    eras = document.get("eras") if isinstance(document, dict) else None
    if not isinstance(eras, list) or not eras:
        raise EraDerivationError(f"era registry {target} has no eras list")
    seen: set[str] = set()
    for row in eras:
        if not isinstance(row, dict) or not row.get("id") or not row.get("scope"):
            raise EraDerivationError(f"era registry {target} has a row with no id/scope")
        if row["id"] in seen:
            raise EraDerivationError(f"era registry {target}: duplicate id {row['id']!r}")
        seen.add(row["id"])
    return eras


def _scoped(eras: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    rows = [row for row in eras if row.get("scope") == scope]
    if not rows:
        raise EraDerivationError(f"era registry has no {scope!r} row")
    return rows


def derive_era(scope: str, at: Any, eras: list[dict[str, Any]] | None = None) -> str:
    """The era id of ``scope`` in force at ``at``. Fails closed."""
    rows = _scoped(eras if eras is not None else load_registry(), scope)
    instant = _instant(at, field="at")
    live = [
        (_instant(row["from"], field="from", era_id=row["id"]), row["id"])
        for row in rows
        if row.get("from") is not None
    ]
    live = [(start, era_id) for start, era_id in live if start <= instant]
    if not live:
        raise EraDerivationError(
            f"no {scope!r} era covers {instant.isoformat()} — the registry starts later"
        )
    return max(live)[1]


def derive_kernel_era(at: Any, eras: list[dict[str, Any]] | None = None) -> str:
    """The KERNEL era in force at ``at``: latest cpu_bench row WITH a binary_version.

    Scope alone is not sufficient — see the module docstring. Raises rather than
    falling back to a neighbouring era when nothing witnesses the instant.
    """
    rows = _scoped(eras if eras is not None else load_registry(), SCOPE_CPU_KERNEL)
    instant = _instant(at, field="at")
    witnessed = [
        (_instant(row["from"], field="from", era_id=row["id"]), row["id"])
        for row in rows
        if row.get("from") is not None and row.get("binary_version") is not None
    ]
    live = [(start, era_id) for start, era_id in witnessed if start <= instant]
    if not live:
        earliest = min(witnessed)[0].isoformat() if witnessed else "never"
        raise EraDerivationError(
            f"no cpu_bench era with a recorded binary_version covers "
            f"{instant.isoformat()}. The earliest witnessed kernel boundary is "
            f"{earliest}. Rows before it (E0, E1, E5-cpu-kernel) record no binary "
            f"version, so no kernel can be named for this instant. REFUSING rather "
            f"than picking a neighbouring era — a stamp naming an instrument nothing "
            f"witnessed is the defect this derivation exists to prevent."
        )
    return max(live)[1]


def era_for_binary(binary_version: Any, eras: list[dict[str, Any]] | None = None) -> str:
    """The cpu_bench era whose recorded binary_version IS ``binary_version``.

    The strongest binding available: it resolves from the witness of what
    executed, not from a clock. Accepts an int or a string containing one (the
    attestation records e.g. ``"version: 10107 (67a433bf4)\\nbuilt with ..."``).
    """
    rows = _scoped(eras if eras is not None else load_registry(), SCOPE_CPU_KERNEL)
    version = _coerce_binary_version(binary_version)
    matches = [row["id"] for row in rows if row.get("binary_version") == version]
    if not matches:
        known = sorted(
            row["binary_version"] for row in rows if row.get("binary_version") is not None
        )
        raise EraDerivationError(
            f"no cpu_bench era records binary_version {version}. Known: {known}. "
            f"REFUSING to stamp a run whose executed binary the registry cannot name — "
            f"either the kernel is unregistered or the attestation is wrong, and both "
            f"need a human."
        )
    if len(matches) > 1:
        raise EraDerivationError(
            f"binary_version {version} is claimed by {matches} — ambiguous, refusing"
        )
    return matches[0]


def _coerce_binary_version(value: Any) -> int:
    """Pull the integer build number out of an attestation field."""
    if isinstance(value, bool):
        raise EraDerivationError("binary_version must not be a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        import re

        match = re.search(r"(?:version:\s*)?(\d{3,})", value)
        if match:
            return int(match.group(1))
    raise EraDerivationError(f"cannot read a binary version out of {value!r}")


def known_era_ids(scope: str, eras: list[dict[str, Any]] | None = None) -> set[str]:
    """Every era id ever recorded for ``scope``. For validating historical stamps."""
    return {row["id"] for row in _scoped(eras if eras is not None else load_registry(), scope)}
