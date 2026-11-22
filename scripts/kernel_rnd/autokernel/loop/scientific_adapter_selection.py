"""AKU-07h5: native-v2 scientific adapter configuration carried by startup manifests.

Selection v1 (correctness = native T0 only) is unchanged and handled by
standalone_inputs. Selection v2 additionally names the server T0 correctness adapter
and the concrete observation witnesses for the ``contention`` / ``residency`` slots,
so a startup manifest can configure the native-v2 (registry v3) adapter set that the
runtime installs. Purpose stays None: it is never an installed adapter.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_V2 = "epyc.autokernel.scientific_adapter_selection.v2"
FIELDS = {"schema", "correctness", "purpose", "contention", "residency"}


class SelectionRefused(ValueError):
    pass


def installed(selection: Mapping[str, Any]):
    """Construct the closed installed registry named by a v2 selection."""
    from . import native_observation_witnesses as now
    from . import native_scientific_witness as nsw
    from . import native_server_t0_witness as server
    row = dict(selection)
    if set(row) != FIELDS or row["schema"] != SCHEMA_V2 or row["purpose"] is not None:
        raise SelectionRefused("scientific adapter selection v2 is unsupported")
    correctness = None
    if row["correctness"] is not None:
        item = row["correctness"]
        if not isinstance(item, Mapping) or set(item) != {"adapter_id", "max_units"}:
            raise SelectionRefused("correctness adapter selection fields differ")
        kinds = {nsw.ADAPTER_ID: nsw.NativeT0WitnessAdapter,
                 server.ADAPTER_ID: server.NativeServerT0WitnessAdapter}
        if item["adapter_id"] not in kinds:
            raise SelectionRefused("configured correctness adapter is not installed")
        correctness = kinds[item["adapter_id"]](max_units=item["max_units"])
    slots = {}
    for name in ("contention", "residency"):
        item = row[name]
        if item is None:
            slots[name] = None
        elif item != {"adapter_id": now.WITNESSES[name]}:
            raise SelectionRefused(f"configured {name} adapter is not installed")
        else:
            slots[name] = now.ObservationWitnessAdapter(name)
    return nsw.ParentScientificWitnessAdapters(correctness=correctness, **slots)
