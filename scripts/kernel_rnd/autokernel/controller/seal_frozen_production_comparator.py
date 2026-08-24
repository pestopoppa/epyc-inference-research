#!/usr/bin/env python3
"""Seal or validate the canonical frozen-v9 AutoKernel comparator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import discovery_deployment_factory as factory


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--validate-only", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output
    factory._validate_comparator_output_path(output)
    if args.validate_only:
        observed = factory._load_frozen_production_comparator(output)
        expected = factory.derive_frozen_production_comparator()
        if observed != expected:
            raise factory.DeploymentFactoryError(
                "comparator differs from current frozen authority")
    else:
        observed = factory.seal_frozen_production_comparator(output)
    print(json.dumps({
        "status": "validated" if args.validate_only else "sealed",
        "path": str(output), "receipt_sha256": observed.receipt_sha256,
        "inference_executed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
