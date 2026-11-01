"""Refresh the deterministic native-control receipt (not a model evaluation)."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .fixtures import generate_fixtures
from .scorer import score

OUTPUT = Path(__file__).parent / "fixtures" / "native_control_proof.json"


def capture() -> dict:
    rows = []
    for fixture in generate_fixtures():
        for control_kind, candidate in (
            ("correct-inverse-control", fixture.expected_source),
            ("mutated-source-negative-control", fixture.mutated_source),
        ):
            row = score(fixture.fixture_id, candidate)
            row["control_kind"] = control_kind
            rows.append(row)
    return {
        "schema": "cj7a_native_control_proof.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "scope": "deterministic native controls only; no model inference or efficacy claim",
        "execution": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "records": rows,
    }


def main() -> int:
    OUTPUT.write_text(json.dumps(capture(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
