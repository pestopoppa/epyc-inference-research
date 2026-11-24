"""Plan-validation errors and field validators for experiment_plan.py, split verbatim from experiment_plan.py (code-budget split)."""
from __future__ import annotations

from .field_validation import integer_field as _integer_field, text_field as _text_field

import math
import re
from types import MappingProxyType
from typing import Any, Mapping, cast

from .. import schemas
from .immutable_values import thaw as _thaw
from .field_validation import exact as _shared_exact
from .field_validation import text_array as _text_array

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PlanValidationError(ValueError):
    """A versioned plan/evidence object violates its declared schema."""


class UnsupportedStoppingRule(PlanValidationError):
    """The stopping rule is well identified but has no implementation here."""


def _exact(obj: Mapping[str, Any], fields: set[str], label: str) -> None:
    return _shared_exact(obj, fields, label, PlanValidationError)


def _text(value: Any, label: str, *, nullable: bool = False) -> str | None:
    return _text_field(value, label, PlanValidationError, nullable=nullable, wording=': expected ')


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    return _integer_field(value, label, PlanValidationError, minimum=minimum, wording=': expected ')


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanValidationError(f"{label}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PlanValidationError(f"{label}: expected finite number")
    return result


def _digest(value: Any, label: str) -> str:
    value = _text(value, label)
    value = cast(str, value)
    if not _SHA256.fullmatch(value):
        raise PlanValidationError(f"{label}: expected lowercase SHA-256")
    return value


def _enum(value: Any, choices: frozenset[str], label: str) -> str:
    value = _text(value, label)
    value = cast(str, value)
    if value not in choices:
        raise PlanValidationError(f"{label}: unsupported {value!r}")
    return value


def _freeze(value: Any, label: str) -> Any:
    """Validate canonical JSON data and recursively make it immutable."""
    try:
        schemas.canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(f"{label}: {exc}") from exc
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item, f"{label}.{key}")
                                 for key, item in value.items()})
    if isinstance(value, list) or isinstance(value, tuple):
        return tuple(_freeze(item, f"{label}[]") for item in value)
    return value



def _strings(value: Any, label: str, *, allow_empty: bool = False,
             unique: bool = True) -> tuple[str, ...]:
    return _text_array(value, label, PlanValidationError, _text, empty=allow_empty, unique=unique)
