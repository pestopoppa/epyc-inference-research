"""AKU-07o2: held GPU device -> KFD gpu_id mapping for the foreign-consumer census.

Device names are the llama.cpp/ROCm enumeration names (``ROCm<i>``, or a bare index).
The i-th device is the i-th KFD GPU node (``gpu_id != 0``) ordered by
``drm_render_minor``. When the topology is unreadable or does not cover a held device,
the startup manifest's explicit ``{device: gpu_id}`` map is the fallback; a device
covered by neither refuses startup. A host with no readable KFD topology and no
manifest map installs no reader, so the census stays honestly unknown.
The map travels in the driver config and the prepared stage to the worker, which
installs ``lifecycle_observation.kfd_consumer_reader`` with it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

TOPOLOGY_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")
_MAX_NODES, _MAX_BYTES = 256, 64 * 1024


class KfdTopologyRefused(ValueError):
    pass


def _properties(path: Path) -> dict[str, str]:
    raw = path.read_bytes()[:_MAX_BYTES].decode("ascii", "replace")
    return dict(line.split(None, 1) for line in raw.splitlines() if len(line.split(None, 1)) == 2)


def topology_gpu_ids(root: Path = TOPOLOGY_ROOT) -> list[str]:
    """KFD gpu_ids of GPU nodes ordered by drm_render_minor; [] when unreadable."""
    try:
        nodes = sorted(Path(root).iterdir())[:_MAX_NODES]
        rows = []
        for node in nodes:
            gpu_id = (node / "gpu_id").read_text().strip()
            if gpu_id.isdigit() and int(gpu_id) != 0:
                minor = int(_properties(node / "properties").get("drm_render_minor", "-1"))
                rows.append((minor, gpu_id))
    except (OSError, ValueError):
        return []
    return [gpu_id for _minor, gpu_id in sorted(rows)]


def validate_map(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if (not isinstance(value, Mapping) or not value
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v.isdigit()
                   for k, v in value.items())):
        raise KfdTopologyRefused("KFD gpu_id map must be a nonempty {device: digits} map")
    return {key: value[key] for key in sorted(value)}


def gpu_id_map(devices: Sequence[str], *, root: Path = TOPOLOGY_ROOT,
               manifest: Mapping[str, str] | None = None) -> dict[str, str] | None:
    """Map each held device to its gpu_id: topology first, manifest fallback."""
    if not devices:
        return None
    fallback = validate_map(manifest) or {}
    ordered = topology_gpu_ids(root)
    if not ordered and not fallback:
        return None  # no KFD on this host and no manifest map: census stays unknown
    result = {}
    for device in devices:
        match = re.fullmatch(r"(?:ROCm)?(\d+)", device)
        if match and int(match.group(1)) < len(ordered):
            result[device] = ordered[int(match.group(1))]
        elif device in fallback:
            result[device] = fallback[device]
        else:
            raise KfdTopologyRefused(f"held GPU device {device!r} has no KFD gpu_id")
    return validate_map(result)


def reader_for(prepared: Any):
    """The worker's KFD consumer reader for a prepared stage, or None when unmapped."""
    mapping = getattr(prepared, "kfd_gpu_ids", None)
    if not mapping:
        return None
    from . import lifecycle_observation as lo
    return lo.kfd_consumer_reader(dict(mapping))
