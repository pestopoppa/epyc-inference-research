"""Installed scientific adapter enumeration, split verbatim from standalone_inputs.py (code-budget split)."""
from __future__ import annotations
import json
from typing import Any, Mapping


SCIENTIFIC_SELECTION_SCHEMA = "epyc.autokernel.scientific_adapter_selection.v1"


class StandaloneInputsRefused(RuntimeError):
    """Startup input is malformed, stale, foreign, or unavailable."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise StandaloneInputsRefused("startup manifest is not canonical JSON") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise StandaloneInputsRefused(f"{label} must be an object with string keys")
    _canonical(value)
    return value


def _installed_scientific_adapters(value: Any):
    """Construct one closed installed adapter registry without restoring issuance."""
    from . import native_scientific_witness as nsw
    selection = dict(_mapping(value, "scientific adapter selection"))
    from . import scientific_adapter_selection as v2  # AKU-07h5: native-v2 registry
    if selection.get("schema") == v2.SCHEMA_V2:
        try:
            return v2.installed(selection)
        except Exception as exc:
            raise StandaloneInputsRefused(f"scientific adapter selection is invalid: {exc}") from exc
    if set(selection) != {"schema", "correctness", "purpose", "contention", "residency"} \
            or selection["schema"] != SCIENTIFIC_SELECTION_SCHEMA \
            or any(selection[name] is not None for name in ("purpose", "contention", "residency")):
        raise StandaloneInputsRefused("scientific adapter selection is unsupported")
    correctness = dict(_mapping(selection["correctness"], "correctness adapter selection"))
    if set(correctness) != {"adapter_id", "max_units"} \
            or correctness["adapter_id"] != nsw.ADAPTER_ID:
        raise StandaloneInputsRefused("configured correctness adapter is not installed")
    try:
        return nsw.ParentScientificWitnessAdapters(
            correctness=nsw.NativeT0WitnessAdapter(max_units=correctness["max_units"]))
    except Exception as exc:
        raise StandaloneInputsRefused(f"scientific adapter selection is invalid: {exc}") from exc
