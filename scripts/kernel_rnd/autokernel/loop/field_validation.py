"""Shared schema field checks with caller-specific validation errors."""
import json
from typing import Any, Mapping


def exact(obj: Mapping[str, Any], fields: set[str], label: str, error: type[ValueError]) -> None:
    if not isinstance(obj, Mapping):
        raise error(f"{label}: expected object")
    missing, extra = fields - set(obj), set(obj) - fields
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(sorted(missing)))
        if extra:
            parts.append("unknown " + ", ".join(sorted(extra)))
        raise error(f"{label}: {'; '.join(parts)}")


def canonical_json(value: Any, error: type[Exception], message: str, *, detail: bool = False) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise error(f"{message}: {exc}" if detail else message) from exc


def require_sha256(value: str, label: str, error: type[Exception], *, article: str = "") -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise error(f"{label} must be {article}lowercase SHA-256")
    return value


def text_array(value: Any, label: str, error: type[Exception], text, *, empty: bool, unique: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise error(f"{label}: expected array")
    items = tuple(text(item, f"{label}[]") for item in value)
    if not empty and not items:
        raise error(f"{label}: must not be empty")
    if unique and len(set(items)) != len(items):
        raise error(f"{label}: duplicate identifiers")
    return items


def text_field(value: Any, label: str, error: type[Exception], *, nullable: bool, wording: str) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value.strip():
        raise error(f"{label}{wording}non-empty text")
    return value


def integer_field(value: Any, label: str, error: type[Exception], *, minimum: int, wording: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise error(f"{label}{wording}integer >= {minimum}")
    return value


def unique_texts(value: Any, label: str, error: type[Exception], text) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise error(f"{label} must be an array")
    result = tuple(text(item, f"{label}[]") for item in value)
    if len(result) != len(set(result)):
        raise error(f"{label} contains duplicates")
    return tuple(str(item) for item in result)
