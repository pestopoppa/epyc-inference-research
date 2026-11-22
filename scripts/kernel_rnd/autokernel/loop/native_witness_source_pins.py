"""Model source pin extraction for native scientific witnesses, split verbatim from native_scientific_witness.py (code-budget split)."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from ..evaluator import c3_epyc_tensor_capture as tensor_capture
from . import lifecycle_observation as lo
from . import observation_binding as ob


class ScientificWitnessRefused(ValueError):
    pass


def _file_stats(paths: tuple[str, ...]) -> tuple[Mapping[str, Any], ...]:
    """Continuity with an original hash receipt; these fields are NOT content hashes."""
    rows = []
    for name in paths:
        path = Path(name)
        info = path.lstat()
        if path.is_symlink() or not path.is_file():
            raise ScientificWitnessRefused("verified inventory member is no longer a regular file")
        rows.append({"path": name, "dev": info.st_dev, "ino": info.st_ino,
            "size": info.st_size, "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
            "mode": info.st_mode, "nlink": info.st_nlink})
    return tuple(ob._freeze(row) for row in rows)


def _model_source_pins() -> Mapping[str, Any]:
    return ob._freeze({"module": lo.prepare_artifact_identity(Path(tensor_capture.__file__)),
        "callables": [lo.callable_identity(function) for function in (
            tensor_capture.CaptureModelIdentity.validate, tensor_capture._checked_regular_file,
            tensor_capture._sha256_file, _file_stats)]})
