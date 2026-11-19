"""AK-AMDAHL-PRUNE: a per-candidate end-to-end ceiling from the serving-profile share.

For a candidate whose `target_symbol` names kernels holding share `s` of the served step:
- zero-time ceiling: the kernel takes no time at all, e2e gain = 1/(1-s) - 1 = s/(1-s);
- realistic ceiling at an X-times op speedup: e2e gain = 1/(1 - s + s/X) - 1.

A candidate whose ceiling is below the keep bar cannot pass the keep gate however good
its op win is. Modes (`--amdahl-prune`):
- `off`: nothing computed;
- `advisory` (default): the ceiling is computed, logged on every candidate record and
  shown to the planner; timing order is unchanged;
- `demote`: as advisory, and candidates whose ceiling is below the bar are timed after
  every other waiting candidate. Never deleted, never refused.

Ranking/order only: the keep gate is unchanged (operator ruling 2026-10-20). An unknown
share (no profile, or the symbol matches no profiled kernel) is `unknown`, never demoted.
The share comes from `serving_shares.py` (AKX-ALL-16 data file).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from . import serving_shares

MODES = ("off", "advisory", "demote")
DEFAULT_MODE = "advisory"
#: Today's keep bar for a GPU e2e lever (3% e2e; candidates under it were killed by hand).
DEFAULT_KEEP_BAR = 0.03
#: A realistic op-level speedup for a kernel rewrite (op 2x is already a strong result).
DEFAULT_SPEEDUP = 2.0
LOG_NAME = "amdahl-ceilings.jsonl"


def ceiling_zero(share: float) -> float:
    """Best-case e2e gain if the kernel's time went to zero."""
    if not 0 <= share < 1:
        raise ValueError(f"share {share} must be in [0, 1)")
    return share / (1.0 - share)


def ceiling_at(share: float, speedup: float) -> float:
    """E2E gain if the kernel ran `speedup` times faster (Amdahl)."""
    if not 0 <= share <= 1 or speedup <= 0:
        raise ValueError("share in [0, 1] and speedup > 0 required")
    return 1.0 / (1.0 - share + share / speedup) - 1.0


def assess(share: float | None, *, keep_bar: float = DEFAULT_KEEP_BAR,
           speedup: float = DEFAULT_SPEEDUP) -> dict[str, Any]:
    """`verdict`: `unknown` (no share), `below_bar_hard` (even zero time < bar),
    `below_bar_realistic` (zero-time clears the bar, X-times does not), `clears`."""
    if share is None:
        return {"share": None, "verdict": "unknown", "keep_bar": keep_bar, "speedup": speedup}
    share = min(max(float(share), 0.0), 0.999999)
    zero, at_x = ceiling_zero(share), ceiling_at(share, speedup)
    verdict = ("below_bar_hard" if zero < keep_bar
               else "below_bar_realistic" if at_x < keep_bar else "clears")
    return {"share": round(share, 6), "ceiling_zero": round(zero, 6),
            "ceiling_at_speedup": round(at_x, 6), "speedup": speedup,
            "keep_bar": keep_bar, "verdict": verdict}


def demoted(record: Mapping[str, Any]) -> bool:
    return record.get("verdict") in {"below_bar_hard", "below_bar_realistic"}


class Assessor:
    """Bound to one serving profile; assesses candidates and appends each to a log."""

    def __init__(self, *, mode: str = DEFAULT_MODE, profile: Mapping[str, Any] | None,
                 profile_id: str | None, keep_bar: float = DEFAULT_KEEP_BAR,
                 speedup: float = DEFAULT_SPEEDUP, log_path: Path | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"amdahl mode {mode!r} not in {MODES}")
        if keep_bar <= 0 or speedup <= 1:
            raise ValueError("keep_bar > 0 and speedup > 1 required")
        self.mode, self.profile, self.profile_id = mode, profile, profile_id
        self.keep_bar, self.speedup, self.log_path = keep_bar, speedup, log_path
        self._lock = threading.Lock()

    def record(self, target_symbol: str | None, *, mechanism_id: str | None = None,
               log: bool = True) -> dict[str, Any] | None:
        if self.mode == "off":
            return None
        row = assess(serving_shares.share_for(self.profile, target_symbol),
                     keep_bar=self.keep_bar, speedup=self.speedup)
        row.update({"mode": self.mode, "profile": self.profile_id,
                    "target_symbol": target_symbol, "mechanism_id": mechanism_id,
                    "demoted": self.mode == "demote" and demoted(row)})
        if self.profile is None:
            row["reason"] = "no serving-shares profile selected"
        if log and self.log_path is not None:
            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as handle:
                    handle.write(json.dumps({"at": time.time(), **row}, sort_keys=True) + "\n")
        return row

    def demote(self, hypothesis: Any) -> bool:
        """Tail hook: True when this candidate is timed after the others."""
        if self.mode != "demote":
            return False
        row = self.record(getattr(hypothesis, "target_symbol", None),
                          mechanism_id=getattr(hypothesis, "mechanism_id", None), log=False)
        return bool(row and row["demoted"])

    def planner_view(self, limit: int = 10) -> dict[str, Any]:
        """Per-kernel-family ceilings for the planner prompt."""
        if self.mode == "off":
            return {"status": "off"}
        if self.profile is None:
            return {"status": "unavailable", "mode": self.mode,
                    "reason": "no serving-shares profile selected"}
        rows = []
        for family, share in serving_shares.family_shares(self.profile)[:limit]:
            if family == serving_shares.OTHER:
                continue
            rows.append({"kernel": family, **assess(share, keep_bar=self.keep_bar,
                                                    speedup=self.speedup)})
        return {"status": "observed", "mode": self.mode, "profile": self.profile_id,
                "keep_bar": self.keep_bar, "speedup": self.speedup, "kernels": rows}


def render(view: Mapping[str, Any]) -> list[str]:
    if view.get("status") != "observed":
        if view.get("status") == "off":
            return []
        return ["", f"## End-to-end ceilings (AK-AMDAHL-PRUNE): {view.get('status')} "
                    f"-- {view.get('reason', '')}"]
    bar, x = view["keep_bar"], view["speedup"]
    lines = ["", f"## End-to-end ceilings by kernel (serving profile `{view['profile']}`)",
             f"Max e2e gain if the kernel went to zero time = s/(1-s); at a {x:g}x op speedup "
             f"= 1/(1-s+s/{x:g})-1. A candidate whose ceiling is below the {bar:.1%} keep bar "
             "cannot pass the keep gate; prefer targets that clear it "
             f"(mode `{view['mode']}`; the keep gate itself is unchanged).",
             "| share | zero-time ceiling | ceiling at speedup | verdict | kernel |",
             "|---|---|---|---|---|"]
    for row in view["kernels"]:
        lines.append(f"| {row['share']:.1%} | {row['ceiling_zero']:.1%} | "
                     f"{row['ceiling_at_speedup']:.1%} | {row['verdict']} | `{row['kernel']}` |")
    return lines


def from_args(args: Any, *, store: Path | None) -> Assessor:
    mode = getattr(args, "amdahl_prune", DEFAULT_MODE) or DEFAULT_MODE
    pid = getattr(args, "serving_shares_profile", None)
    profile = None
    if mode != "off" and pid:
        shares = serving_shares.load(getattr(args, "serving_shares", None))
        profile = serving_shares.get_profile(shares, pid)
        if profile is None:
            raise serving_shares.SharesInvalid(f"serving-shares profile {pid!r} not in file")
    return Assessor(mode=mode, profile=profile, profile_id=pid,
                    keep_bar=float(getattr(args, "amdahl_keep_bar", DEFAULT_KEEP_BAR)),
                    speedup=float(getattr(args, "amdahl_speedup", DEFAULT_SPEEDUP)),
                    log_path=(Path(store) / LOG_NAME) if store is not None else None)


def add_arguments(parser) -> None:
    parser.add_argument("--amdahl-prune", choices=MODES, default=DEFAULT_MODE,
                        help="AK-AMDAHL-PRUNE: per-candidate e2e ceiling from the serving "
                             "share; advisory logs and shows it, demote also times "
                             "below-bar candidates last; never refuses a candidate")
    parser.add_argument("--serving-shares", type=Path, default=None,
                        help="AKX-ALL-16 serving-shares JSON (default: the packaged "
                             "data/gpu_serving_shares_v1.json)")
    parser.add_argument("--serving-shares-profile", default=None,
                        help="profile id <model>/<mode>, e.g. 27b-q8_0/dflash2_verify_d32768")
    parser.add_argument("--amdahl-keep-bar", type=float, default=DEFAULT_KEEP_BAR)
    parser.add_argument("--amdahl-speedup", type=float, default=DEFAULT_SPEEDUP)


__all__ = ["Assessor", "DEFAULT_KEEP_BAR", "DEFAULT_MODE", "DEFAULT_SPEEDUP", "MODES",
           "add_arguments", "assess", "ceiling_at", "ceiling_zero", "demoted",
           "from_args", "render"]
