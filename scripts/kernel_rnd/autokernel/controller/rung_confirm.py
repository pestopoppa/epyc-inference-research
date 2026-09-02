#!/usr/bin/env python3
"""The confirm half of the two-rung screen/confirm keep gate (D1-D6, 2026-09-01).

Spec: docs/design/autokernel-production-shaped-rung.md §5.3, operator-approved as
Option C. R23-5 is the incident this closes: the champion's +17.26% screen-rung
headline (1.5B, ne11=1) was +3.83% at ne11=2 and a decisive -1.46% REGRESSION at
ne11=8 on the production shape -- a keep gate that only ever sees the screen rung
cannot see that inversion, and the standing headline inflated for weeks.

THE SHAPE. A screen-rung keep is a KEEP_CANDIDATE, not a keep. Promotion to `kept`
is one extra `bench.compare` per confirm surface (D2: dec-b4 + dec-b8; D3: pairs=5,
the calibrated k=5 floor row) on the production-shaped confirm model, run inside the
same serialized tail as the commit -- no second harness, no second claim. The gate
refuses when any confirm surface shows a DECISIVE regression, and -- fail-closed,
the same keep-refusal doctrine as `run.refuse_uncalibrated_keep` -- when a confirm
surface is UNCALIBRATED for the confirm model: promoting through a floor nobody
measured is how fake-decisive keeps were manufactured once already (§5.2).

Both dispositions write the full record (screen comparison, every confirm
comparison, both rung-parity records) into `<store>/confirm/`, so a vetoed
candidate is evidence anyone can re-read, not a vanished measurement.

This module is deliberately in `controller/` (the uncounted library): the loop's
budgeted footprint carries only the wiring -- flags, one closure, one call at the
keep gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import workload_contract

CONFIRM_SCHEMA = "epyc.autokernel.confirm_gate.v1"
#: D2 -- dec-b4 (the operator's primary) + dec-b8 (the R23-5 inversion class).
DEFAULT_SURFACES = ("dec-b4", "dec-b8")
#: D3 -- the calibrated k=5 floor row; 20 pairs is reserved for the standing
#: champion-vs-production headline refresh, never the per-candidate gate.
DEFAULT_PAIRS = 5


def _write_record(store: Path, name: str, body: Mapping[str, Any]) -> Path:
    """Atomic single-document publish, the same contract as `status.write_json`
    (not imported: `controller` does not depend on the loop package)."""
    store.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(store), prefix=".confirm-")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(body, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, store / name)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return store / name


@dataclass(frozen=True)
class Confirm:
    """A configured confirm rung. Built once at startup by `configure`."""

    model: Path
    pairs: int
    surfaces: tuple[str, ...]
    store: Path
    floors: Mapping[str, float | None]
    screen_parity: workload_contract.RungParity
    confirm_parity: workload_contract.RungParity

    def describe(self) -> str:
        floors = ", ".join(
            f"{surface} {'%.3f%%' % value if value is not None else 'UNCALIBRATED'}"
            for surface, value in ((s, self.floors.get(s)) for s in self.surfaces))
        waiver = (" — screen parity WAIVED (recorded)" if self.screen_parity.waived
                  else "")
        return (f"{self.model.name}, {self.pairs} pairs on "
                f"{'+'.join(self.surfaces)} ({floors}){waiver}")

    def gate(self, mechanism_id: str, screen: Any,
             measure: Callable[[str, float | None], Any]) -> dict[str, Any]:
        """Measure the keep-candidate on every confirm surface and decide.

        `measure(surface, floor_pct) -> bench.Comparison` is injected (run.py binds
        the arms, the confirm model and the pair count), so the gate is exercised
        with no GPU. Returns the persisted record; `record["promoted"]` False means
        the candidate stays a KEEP_CANDIDATE and the caller must not commit it.
        """
        rows = [measure(surface, self.floors.get(surface))
                for surface in self.surfaces]
        vetoes: list[str] = []
        for row in rows:
            if row.decisive is None:
                # Fail-closed, exactly like the screen's own uncalibrated refusal:
                # a confirm gate that waves candidates through an unmeasured floor
                # is not a gate, it is the fake-decisive defect with a second rung.
                vetoes.append(
                    f"{row.surface}: UNCALIBRATED on {self.model.name} — run "
                    f"--calibrate-surface for the confirm model before enabling "
                    f"promotion through it")
            elif row.decisive and row.effect < 0:
                vetoes.append(
                    f"{row.surface}: decisive regression {row.effect * 100:+.3f}% "
                    f"on the production shape (floor {row.noise_floor_pct}%) — the "
                    f"R23-5 inversion class this rung exists to catch")
        promoted = not vetoes
        reason = ("confirmed on the production shape: " + ", ".join(
                      f"{row.surface} {row.effect * 100:+.3f}%" for row in rows)
                  if promoted else
                  "confirm rung vetoed the keep: " + " | ".join(vetoes))
        record = {
            "schema": CONFIRM_SCHEMA,
            "mechanism_id": mechanism_id,
            "model": str(self.model),
            "pairs": self.pairs,
            "surfaces": list(self.surfaces),
            "promoted": promoted,
            "reason": reason,
            "screen": screen.to_dict(),
            "confirm": [row.to_dict() for row in rows],
            "parity": {"screen": self.screen_parity.to_dict(),
                       "confirm": self.confirm_parity.to_dict()},
            "recorded_at": datetime.now(timezone.utc).isoformat()
                                   .replace("+00:00", "Z"),
        }
        stamp = record["recorded_at"].replace(":", "").replace("-", "")
        _write_record(Path(self.store) / "confirm",
                      f"{mechanism_id}.{stamp}.json", record)
        return record


def configure(*, model: Path | str, pairs: int, surfaces: Sequence[str] | str,
              store: Path, screen_census: workload_contract.WorkloadCensus,
              known_surfaces: Sequence[str],
              floor_for: Callable[[str], float | None],
              production_model: Path | str = workload_contract.PRODUCTION_MODEL
              ) -> Confirm:
    """Build the confirm rung, refusing a misconfigured one at STARTUP.

    A confirm model that is not production-shaped is refused outright
    (`rung_matches_production`, exact required): a non-exact confirm rung is the
    screen's job done twice and the confirm's not at all. The screen's own parity
    is computed here too and rides on every gate record -- that is the §5.1 waiver
    artifact.
    """
    model = Path(model)
    names = (tuple(part.strip() for part in surfaces.split(",") if part.strip())
             if isinstance(surfaces, str) else tuple(surfaces))
    if not names:
        raise workload_contract.WorkloadContractError(
            "the confirm rung was configured with no surfaces; a gate over nothing "
            "promotes everything")
    unknown = [name for name in names if name not in known_surfaces]
    if unknown:
        raise workload_contract.WorkloadContractError(
            f"unknown confirm surface(s) {unknown}: the loop drives "
            f"{sorted(known_surfaces)}")
    if pairs < 1:
        raise workload_contract.WorkloadContractError(
            f"--confirm-pairs {pairs}: a comparison needs at least one pair")
    production = workload_contract.production_census(production_model)
    confirm_census = workload_contract.verify_workload(model, production=production)
    confirm_parity = workload_contract.rung_matches_production(
        confirm_census, production, rung=workload_contract.CONFIRM_RUNG)
    if not confirm_parity.exact:
        raise workload_contract.WorkloadContractError(
            f"--confirm-model {model} is not production-shaped: "
            f"{confirm_parity.detail}")
    screen_parity = workload_contract.rung_matches_production(
        screen_census, production, rung=workload_contract.SCREEN_RUNG)
    return Confirm(model=model, pairs=pairs, surfaces=names, store=Path(store),
                   floors={name: floor_for(name) for name in names},
                   screen_parity=screen_parity, confirm_parity=confirm_parity)


__all__ = ["CONFIRM_SCHEMA", "Confirm", "DEFAULT_PAIRS", "DEFAULT_SURFACES",
           "configure"]
