"""AKX-P0d/P1d: the per-workload census STORE, its freshness keys, and live T0/T1 witnesses.

`census.py` produces and writes `workload_census.v1` documents; nothing created the store
the keep hook reads or derived the expected freshness key, so production never classified
INERT (`record_keep_best_effort` passed `censuses=None`). This module is that glue:

* ``ensure_store`` creates ``DEFAULT_CENSUS_STORE`` (env ``AK_CENSUS_STORE`` overrides).
* ``census_identity`` is the ONE identity a producer must stamp on a census so the hook can
  re-derive the freshness key from live state alone (workload signature + champion commit).
  ``model`` is keyed by the workload signature, never by a path, so a moved GGUF does not
  stale the record while a changed champion or recipe does.
* ``dso_digests`` (T0) hashes the shared libraries of a build; ``t1_profile_sha`` hashes the
  scheduler op/backend profile a census recorded per shape (T1). Witnesses are computed from
  the caller's own anchor and candidate builds/forwards, never read from the census file.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from . import census as CE

STORE_ENV = "AK_CENSUS_STORE"
STORE_SCHEMA = "epyc.autokernel.census_store.v1"


def store_path() -> Path:
    return Path(os.environ.get(STORE_ENV) or CE.DEFAULT_CENSUS_STORE)


def ensure_store(path: Path | str | None = None) -> Path:
    """Create the census store directory and its STORE.json marker (idempotent)."""
    root = Path(path) if path is not None else store_path()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "STORE.json"
    if not marker.exists():
        marker.write_text(json.dumps({
            "schema": STORE_SCHEMA, "file_pattern": "<workload_id>-<freshness_key[:16]>.json",
            "document_schema": CE.SCHEMA,
            "identity": "census_store.census_identity(champion_commit=..., probe=...)",
        }, indent=1, sort_keys=True) + "\n")
    return root


def census_identity(*, champion_commit: str, probe: str | None = None) -> dict[str, Any]:
    """Identity extras a producer passes to `census.produce_census(identity=...)`.

    ``probe`` names a reduced shape envelope (a witness exercise, not the production
    recipe's envelope) so such a record can never satisfy a production-envelope lookup."""
    if not champion_commit:
        raise ValueError("champion_commit is required")
    out: dict[str, Any] = {"champion_commit": champion_commit,
                           "model": {"keyed_by": "workload_signature"}}
    if probe:
        out["probe"] = probe
    return out


def expected_keys(resolved_targets: Sequence[Mapping[str, Any]], *, champion_commit: str,
                  probe: str | None = None) -> dict[str, str]:
    """workload_id -> the freshness key the CURRENT champion would give its census."""
    keys: dict[str, str] = {}
    for target in resolved_targets or ():
        sig = target.get("workload_signature") if isinstance(target, Mapping) else None
        if not isinstance(sig, str):
            continue
        for wid in target.get("target_ids") or ():
            if isinstance(wid, str) and wid:
                keys[wid] = CE.freshness_key({
                    "workload_id": wid, "workload_signature": sig,
                    **census_identity(champion_commit=champion_commit, probe=probe)})
    return keys


def censuses_for_keep(resolved_targets: Sequence[Mapping[str, Any]], *, champion_commit: str,
                      store: Path | str | None = None, probe: str | None = None) -> dict[str, tuple]:
    """``record_keep_screen(censuses=...)`` input from the store; {} when the store is absent
    or holds nothing fresh (every workload then classifies UNKNOWN, i.e. TOUCHED)."""
    from . import blast_radius as BR
    root = Path(store) if store is not None else store_path()
    if not root.is_dir() or not champion_commit:
        return {}
    return BR.load_censuses(root, expected_keys(resolved_targets, champion_commit=champion_commit,
                                                probe=probe))


# --- T0: DSO digests ---------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dso_digests(bin_dir: Path | str) -> dict[str, str]:
    """sha256 of every real ``libggml*.so`` / ``libllama*.so`` in a build's bin dir, keyed by
    the unversioned file name. Symlinks resolve to their target and are not double counted.
    An empty result is not a witness (T0 needs at least one library)."""
    root = Path(bin_dir)
    out: dict[str, str] = {}
    for path in sorted(root.glob("lib*.so*")):
        real = path.resolve()
        if not real.is_file() or not path.name.startswith(("libggml", "libllama")):
            continue
        name = path.name.split(".so")[0] + ".so"
        out.setdefault(name, _sha256_file(real))
    return out


# --- T1: scheduler profile hashes --------------------------------------------------------
def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def t1_profile_sha(census: Mapping[str, Any]) -> str:
    """Hash of the per-shape scheduler op/backend profile a census recorded."""
    rows = (census.get("graph") or {}).get("per_shape") or []
    return hashlib.sha256(_canon([{"shape": r.get("shape"), "op_backend": r.get("op_backend")}
                                  for r in rows]).encode()).hexdigest()


def envelope_sha(census: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canon((census.get("graph") or {}).get("shape_envelope") or []).encode()
                          ).hexdigest()


def t1_witness(anchor: Mapping[str, Any], candidate: Mapping[str, Any]) -> Optional[tuple]:
    """``(anchor_sha, candidate_sha, envelope_sha)`` when BOTH forwards were observed on the
    same shape envelope; None otherwise (no witness is better than a vacuous one)."""
    if CE.census_state(anchor) != CE.OBSERVED or CE.census_state(candidate) != CE.OBSERVED:
        return None
    if envelope_sha(anchor) != envelope_sha(candidate):
        return None
    return (t1_profile_sha(anchor), t1_profile_sha(candidate), envelope_sha(anchor))


def with_witnesses(record, *, t0: tuple[Mapping[str, str], Mapping[str, str]] | None = None,
                   t1: tuple | None = None):
    """A WorkloadRecord carrying the caller's recorded witnesses (None = unchanged)."""
    changes: dict[str, Any] = {}
    if t0 is not None and t0[0] and t0[1]:
        changes.update(t0_dso_anchor=dict(t0[0]), t0_dso_candidate=dict(t0[1]))
    if t1 is not None:
        changes["t1_profiles"] = tuple(t1)
    return dataclasses.replace(record, **changes) if changes else record


__all__ = ["STORE_ENV", "STORE_SCHEMA", "censuses_for_keep", "census_identity", "dso_digests",
           "ensure_store", "envelope_sha", "expected_keys", "store_path", "t1_profile_sha",
           "t1_witness", "with_witnesses"]
