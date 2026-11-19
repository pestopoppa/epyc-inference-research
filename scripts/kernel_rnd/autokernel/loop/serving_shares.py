"""AKX-ALL-16 (G2): versioned GPU serving-profile shares as a planner input.

A serving-shares file holds, per (model, mode) profile, each kernel's share of GPU busy
time in the served window, computed from rocprofv3 kernel traces already on disk. It is
data, not a live profile: `build` turns trace CSVs into a profile entry, `load` validates
a file, `share_for` sums the shares of the kernels a candidate's `target_symbol` names
(same matching rule as `gpu_serving_profile.kernel_weight`). `amdahl.py` consumes it.

Windows:
- `post_prefill`: dispatches that start after the last `Cijk` (hipBLASLt prefill GEMM)
  kernel ends -- the decode / DFlash2 verify window of a llama-bench `-p P -n N` trace
  (the rule of `/mnt/raid0/llm/tmp/c16-scout-work/agg2.py`).
- `all`: every dispatch (llama-bench `-p n` verify-shaped traces with no prefill tail).

Shares are a ranking input only; they never gate a keep (operator ruling 2026-10-20).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .gpu_serving_profile import _kernel_matches, normalize_name

# Rebuild: cd scripts/kernel_rnd && python3 -m autokernel.loop.serving_shares \
#   --spec autokernel/loop/data/gpu_serving_shares_v1.spec.json \
#   --out autokernel/loop/data/gpu_serving_shares_v1.json

SCHEMA = "epyc.autokernel.gpu_serving_shares.v1"
DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "gpu_serving_shares_v1.json"
WINDOW_RULES = frozenset({"post_prefill", "all"})
#: Kernels below this share are folded into `other` (the table stays reviewable).
MIN_SHARE = 0.002
OTHER = "(other)"


class SharesInvalid(ValueError):
    """The serving-shares file does not satisfy the schema."""


def _rows(path: Path) -> list[tuple[int, int, str]]:
    out = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("Kernel_Name") or "").strip()
            try:
                start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if name and end > start:
                out.append((start, end, name))
    out.sort()
    return out


def window_rows(rows: Sequence[tuple[int, int, str]], rule: str) -> list[tuple[int, int, str]]:
    if rule not in WINDOW_RULES:
        raise SharesInvalid(f"window rule {rule!r} unsupported")
    if rule == "all":
        return list(rows)
    last = max((end for _s, end, name in rows if name.startswith("Cijk")), default=0)
    return [row for row in rows if row[0] > last]


def build(traces: Iterable[Path], *, rule: str, model: str, mode: str,
          description: str = "", min_share: float = MIN_SHARE) -> dict[str, Any]:
    """One profile entry from one or more traces of the same (model, mode); durations
    are summed across traces, so repeats are pooled by time."""
    totals: dict[str, int] = {}
    calls: dict[str, int] = {}
    sources, busy, wall = [], 0, 0
    for trace in traces:
        trace = Path(trace)
        raw = trace.read_bytes()
        rows = window_rows(_rows(trace), rule)
        if not rows:
            raise SharesInvalid(f"{trace}: no dispatches in window {rule}")
        trace_busy = sum(end - start for start, end, _n in rows)
        busy += trace_busy
        wall += rows[-1][1] - rows[0][0]
        for start, end, name in rows:
            key = normalize_name(name)
            totals[key] = totals.get(key, 0) + end - start
            calls[key] = calls.get(key, 0) + 1
        sources.append({"path": str(trace), "sha256": hashlib.sha256(raw).hexdigest(),
                        "dispatches": len(rows), "busy_ns": trace_busy})
    if not sources:
        raise SharesInvalid("no traces")
    kernels, other_ns, other_calls = [], 0, 0
    for name in sorted(totals, key=lambda k: (-totals[k], k)):
        share = totals[name] / busy
        if share < min_share:
            other_ns += totals[name]
            other_calls += calls[name]
            continue
        kernels.append({"kernel": name, "share": round(share, 6), "calls": calls[name]})
    if other_ns:
        kernels.append({"kernel": OTHER, "share": round(other_ns / busy, 6),
                        "calls": other_calls})
    return {"model": model, "mode": mode, "window_rule": rule, "description": description,
            "busy_fraction": round(busy / wall, 4) if wall else None,
            "busy_ns": busy, "sources": sources, "kernels": kernels}


def profile_id(entry: Mapping[str, Any]) -> str:
    return f"{entry['model']}/{entry['mode']}"


def validate(body: Any) -> dict[str, Any]:
    if not isinstance(body, Mapping) or body.get("schema") != SCHEMA:
        raise SharesInvalid(f"schema must be {SCHEMA}")
    for field in ("version", "generated", "hardware", "profiles"):
        if field not in body:
            raise SharesInvalid(f"missing {field}")
    if not isinstance(body["profiles"], list) or not body["profiles"]:
        raise SharesInvalid("profiles must be a non-empty array")
    seen = set()
    for entry in body["profiles"]:
        for field in ("model", "mode", "window_rule", "sources", "kernels"):
            if field not in entry:
                raise SharesInvalid(f"profile missing {field}")
        pid = profile_id(entry)
        if pid in seen:
            raise SharesInvalid(f"duplicate profile {pid}")
        seen.add(pid)
        if entry["window_rule"] not in WINDOW_RULES:
            raise SharesInvalid(f"{pid}: window rule unsupported")
        if not entry["sources"]:
            raise SharesInvalid(f"{pid}: no sources (shares must come from traces)")
        total = 0.0
        for row in entry["kernels"]:
            share = row.get("share")
            if not isinstance(share, (int, float)) or not 0 <= share <= 1 or not row.get("kernel"):
                raise SharesInvalid(f"{pid}: bad kernel row {row!r}")
            total += share
        if abs(total - 1.0) > 0.01:
            raise SharesInvalid(f"{pid}: shares sum to {total:.4f}, not 1")
    return dict(body)


def load(path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        body = json.loads(target.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SharesInvalid(f"{target}: {exc}") from exc
    return validate(body)


def get_profile(shares: Mapping[str, Any], pid: str) -> dict[str, Any] | None:
    return next((dict(e) for e in shares.get("profiles") or () if profile_id(e) == pid), None)


def share_for(profile: Mapping[str, Any] | None, target_symbol: str | None) -> float | None:
    """Summed share of the kernels `target_symbol` names; None when nothing matches
    (unknown is not zero: an unmatched symbol is never pruned)."""
    if not profile or not target_symbol:
        return None
    # A fusion candidate may name several kernels: "a|b|c" sums each kernel once.
    symbols = [s.strip() for s in str(target_symbol).split("|") if s.strip()]
    hits = [float(row["share"]) for row in profile.get("kernels") or ()
            if row["kernel"] != OTHER
            and any(_kernel_matches(row["kernel"], sym) for sym in symbols)]
    return round(sum(hits), 6) if hits else None


def family_shares(profile: Mapping[str, Any]) -> list[tuple[str, float]]:
    """Shares summed per base kernel name (text before the template), largest first."""
    out: dict[str, float] = {}
    for row in profile.get("kernels") or ():
        base = row["kernel"].split("<", 1)[0].strip()
        out[base] = out.get(base, 0.0) + float(row["share"])
    return sorted(((k, round(v, 6)) for k, v in out.items()), key=lambda kv: (-kv[1], kv[0]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True,
                        help="JSON: {version, generated, hardware, provenance, profiles: "
                             "[{model, mode, window_rule, description, traces: [...]}]}")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    body = {"schema": SCHEMA, "version": spec["version"], "generated": spec["generated"],
            "hardware": spec["hardware"], "provenance": spec.get("provenance", ""),
            "profiles": [build([Path(t) for t in item["traces"]], rule=item["window_rule"],
                               model=item["model"], mode=item["mode"],
                               description=item.get("description", ""))
                         for item in spec["profiles"]]}
    validate(body)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1, sort_keys=False) + "\n")
    print(f"wrote {args.out}: {len(body['profiles'])} profiles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
