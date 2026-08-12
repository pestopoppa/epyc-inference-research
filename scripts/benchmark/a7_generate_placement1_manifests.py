#!/usr/bin/env python3
"""A7 / T3 — regenerate the confounded E5 Stage-B cells on P-BENCH-PLACEMENT-1 shapes.

The Stage-B grid was run with QUARTER placement arms that no longer exist, and with
placement INHERITED rather than DECLARED — the defect T3 exists to correct. This script
re-emits each confounded cell on the ratified 1-full + 2-half grid.

Placement constants are IMPORTED from e5_cell_manifests, never typed here: a second
spelling of a cpuset is a second source of truth, which is the class of defect that
produced the drift this whole handoff is about.

Emits, per confounded source cell, the shapes the row requires:
  FULL    1 instance  CPUSET_FULL   with numactl_policy=interleave=all  (declared, not inherited)
  2xHALF  2 instances CPUSET_HALF0 + CPUSET_HALF1

Writes nothing outside --out-dir. Does not launch anything.
"""
from __future__ import annotations
import argparse, copy, json, os, sys, glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark.e5_cell_manifests import (  # noqa: E402
    CPUSET_FULL, CPUSET_HALF0, CPUSET_HALF1, CONFIG_INSTANCE_COUNT, K_LADDER,
)

# config_id ENCODES the instance count, so re-shaping a cell REQUIRES remapping it.
# C3 (4 retired quarters) and C2 (2 quarters) both become C1 (1 full) / C1b (2 halves).
# Imported, not hardcoded: CONFIG_INSTANCE_COUNT is the authority for how many
# instances each config means, and the driver enforces it.
ARM_CONFIG = {"full": "C1", "half": "C1b"}

RETIRED_QUARTERS = {"0-23,96-119", "24-47,120-143", "48-71,144-167", "72-95,168-191"}
PLACEMENT_PROTOCOL = "P-BENCH-PLACEMENT-1"


def is_confounded(manifest: dict) -> bool:
    """A cell is confounded iff it declares a Stage-B family AND uses a retired quarter arm."""
    if not manifest.get("stage_b_families"):
        return False
    return any(i.get("cpu_list") in RETIRED_QUARTERS for i in manifest.get("instances") or [])


def _base_port(manifest: dict) -> int:
    inst = manifest.get("instances") or [{}]
    return int(inst[0].get("port", 19380))


def full_shape(manifest: dict) -> list[dict]:
    return [{"cpu_list": CPUSET_FULL, "threads": 96,
             "numactl_policy": "interleave=all", "port": _base_port(manifest)}]


def half_shape(manifest: dict) -> list[dict]:
    p = _base_port(manifest)
    return [{"cpu_list": CPUSET_HALF0, "threads": 96, "numactl_policy": "none", "port": p},
            {"cpu_list": CPUSET_HALF1, "threads": 96, "numactl_policy": "none", "port": p + 1}]


def emit(manifest: dict, arm: str) -> dict:
    out = copy.deepcopy(manifest)
    out["instances"] = full_shape(manifest) if arm == "full" else half_shape(manifest)
    cfg = ARM_CONFIG[arm]
    assert len(out["instances"]) == CONFIG_INSTANCE_COUNT[cfg], (
        f"shape/config disagree: {cfg} means {CONFIG_INSTANCE_COUNT[cfg]} instance(s), "
        f"built {len(out['instances'])}")
    out["config_id"] = cfg
    out["a7_source_config"] = manifest.get("config_id")
    # cell_id must match {model_key}-{config_id}-np{np}[-suffix]; the config moved, so the
    # id must be REBUILT from the new config rather than suffixed onto the old one. The
    # original suffix (e.g. e1parity) is preserved so provenance is not silently dropped.
    mk, np_ = manifest["model_key"], manifest["np"]
    old_prefix = f"{mk}-{manifest.get('config_id')}-np{np_}"
    orig_suffix = manifest["cell_id"][len(old_prefix):].lstrip("-")
    parts = [f"{mk}-{cfg}-np{np_}"] + ([orig_suffix] if orig_suffix else []) + [f"a7{arm}"]
    out["cell_id"] = "-".join(parts)
    out["placement_protocol"] = PLACEMENT_PROTOCOL
    out["placement_declared"] = True
    out["a7_source_cell"] = manifest["cell_id"]
    out["a7_source_shape"] = "quarters(retired)"
    out["notes"] = (manifest.get("notes", "") or "") + (
        f" A7/T3 {PLACEMENT_PROTOCOL} REGENERATION 2026-08-12: source cell ran on retired QUARTER "
        f"arms with placement inherited. Re-emitted on the ratified 1-full + 2-half grid with "
        f"placement DECLARED per instance. Not comparable to the source cell's numbers.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-root", default="data/batched_decode/e5_manifests")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude-group", action="append", default=[],
                    help="model group to skip (e.g. locally-derived duplicates)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    found, written, skipped, converged = [], 0, [], []
    emitted: dict[str, str] = {}   # cell_id -> source cell, to detect convergence
    for f in sorted(glob.glob(os.path.join(a.manifest_root, "*", "*.json"))):
        group = os.path.basename(os.path.dirname(f))
        if group in a.exclude_group:
            continue
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if not is_confounded(j):
            continue
        found.append((group, j["cell_id"]))
        for arm in ("full", "half"):
            cfg = ARM_CONFIG[arm]
            if j.get("np") not in K_LADDER[cfg]:
                skipped.append((group, j["cell_id"], arm, j.get("np"), cfg))
                continue
            out = emit(j, arm)
            prior = emitted.get(out["cell_id"])
            if prior is not None:
                # Two source cells converge on ONE corrected cell. This is SEMANTICALLY
                # CORRECT -- they differed only in a placement dimension that no longer
                # exists -- but it must be reported, never silently overwritten.
                converged.append((out["cell_id"], prior, j["cell_id"]))
                continue
            emitted[out["cell_id"]] = j["cell_id"]
            d = os.path.join(a.out_dir, group)
            if not a.dry_run:
                os.makedirs(d, exist_ok=True)
                json.dump(out, open(os.path.join(d, out["cell_id"] + ".json"), "w"),
                          indent=2, sort_keys=True)
            written += 1

    if not found:
        print("REFUSING: zero confounded cells matched — the selector found nothing to "
              "regenerate, which is a defect in the selector or the corpus, not a clean run.",
              file=sys.stderr)
        return 2

    if skipped:
        print(f"NOT REGENERATED — np outside the target config's K_LADDER ({len(skipped)}):")
        for g, c, arm, np_, cfg in skipped:
            print(f"  {g:30s} {c:34s} arm={arm:4s} np={np_} not in K_LADDER[{cfg}]")
    if converged:
        print(f"CONVERGED — distinct quarter-shaped cells that become ONE corrected cell "
              f"({len(converged)}); the grid was redundant in a dimension that no longer exists:")
        for cid, first, second in converged:
            print(f"  {cid}  <- {first} AND {second}")
    print(f"confounded source cells: {len(found)}")
    print(f"manifests {'would be ' if a.dry_run else ''}written: {written} "
          f"(from {len(found)} source cells x 2 arms, minus {len(converged)} converged)")
    for g, c in found:
        print(f"  {g:30s} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
