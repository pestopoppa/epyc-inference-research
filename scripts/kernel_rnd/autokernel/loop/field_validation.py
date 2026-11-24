"""Shared schema field checks with caller-specific validation errors."""
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
