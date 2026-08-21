# Hawkeye schema provenance

Only two vendor-generic data contracts are derived here from
Zanatticus/Hawkeye commit `a226e955d56c04be044d46f6fd876191cfce5bf4`,
which declares Apache-2.0 in its `pyproject.toml`:

- `hawkeye_tensor_manifest.schema.json` records the `dump_manifest.json`
  structure documented and parsed by `workloads/_tensorset.h`.
- `hawkeye_timing_result.schema.json` records the `result.json` structure
  emitted by `workloads/_driver.cpp`.

AutoKernel did not import Hawkeye's `spec.json` files, architecture registry,
workload leaf layout, hardware taxonomy, or skill library. The executable
hardening and validation code is project-authored in `hawkeye_measurement.py`.
