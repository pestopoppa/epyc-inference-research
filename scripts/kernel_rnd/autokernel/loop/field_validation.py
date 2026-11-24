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
