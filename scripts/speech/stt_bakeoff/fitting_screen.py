"""CS-8.STT7 predictions-only benchmark-fitting screen (reports, never gates).

fitting indicator: HIGHER accept-ref = MORE fitting (the arm reproduces reference errors the
consensus panel disagrees with). It is a screen, not a trap, and `gates` is always False.

This module runs NO subprocess and NO inference: it builds the argv for
`benchmark-optimization ref-disagreement` and `score_voxpopuli_ref_errors.py`, hashes prediction
files, parses per-item decisions and computes accept-ref + Wilson interval.

Per-item decisions schema (JSONL, one object per line) -- the adapter wrapper that runs the
tool maps the tool's `--out` JSON (edits[].verdict = {model: "consensus"|"ref"|null}) to it:
    {"item_id": str, "eligible": bool,
     "decision": "accept_ref" | "accept_hyp" | "other" | "unparseable"}
  accept_ref  = model reproduced the reference (tool verdict "ref")
  accept_hyp  = model sided with the panel consensus (tool verdict "consensus")
  other       = eligible but neither
  unparseable = prediction missing/unreadable; COUNTED in the denominator, never dropped.
Ineligible items (eligible=false, e.g. tool verdict None) are excluded from n_eligible.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# pinned upstream commit of the ref-disagreement tool (dive-asr-benchmark-optimization)
BENCHMARK_OPTIMIZATION_COMMIT = "9b55a563c1356b8fc33ec8199698c54edb7305c1"
ACCEPT_REF_THRESHOLD = 0.10  # strictly above => label
LABEL_UNTRUSTWORTHY = "public WER not trustworthy; private set only"
CONTROL_ARM = "whisper-large-v3"
DECISIONS = ("accept_ref", "accept_hyp", "other", "unparseable")
METRIC_DIRECTION = ("fitting indicator: higher accept-ref = more fitting (reproduces reference errors); "
                    "reports, never gates")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for k successes of n; (0,1) when n == 0."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def parse_decisions(lines: Iterable[str]) -> list[dict]:
    """Parse per-item JSONL. A line that does not parse or has a bad decision becomes an
    `unparseable` eligible item (counted in the denominator, never silently dropped)."""
    out = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            item_id = str(o["item_id"])
            eligible = bool(o["eligible"])
            decision = o["decision"]
            if decision not in DECISIONS:
                raise ValueError(decision)
            out.append({"item_id": item_id, "eligible": eligible, "decision": decision})
        except (ValueError, KeyError, TypeError):
            out.append({"item_id": f"line-{i + 1}", "eligible": True, "decision": "unparseable"})
    return out


def load_decisions(path: str | Path) -> list[dict]:
    return parse_decisions(Path(path).read_text(encoding="utf-8").splitlines())


def accept_ref(items: Sequence[Mapping[str, Any]]) -> dict:
    """accept_ref = accepted-reference / n_eligible; unparseable are IN the denominator."""
    elig = [it for it in items if it.get("eligible")]
    n = len(elig)
    k = sum(1 for it in elig if it["decision"] == "accept_ref")
    n_unp = sum(1 for it in elig if it["decision"] == "unparseable")
    lo, hi = wilson_interval(k, n)
    return {"accept_ref": (k / n) if n else None, "accept_ref_ci": [lo, hi], "n_eligible": n,
            "n_accept_ref": k, "n_unparseable": n_unp}


def label_for(point: float | None) -> str | None:
    """Strictly above 0.10 on the POINT estimate; exactly 0.10 does not label."""
    return LABEL_UNTRUSTWORTHY if (point is not None and point > ACCEPT_REF_THRESHOLD) else None


def build_screen_plan(arms: Sequence[str], predictions_paths: Mapping[str, str | Path], panel_commit: str,
                      panel_models: Sequence[str] = (), out_dir: str | Path = "fitting_screen_out",
                      language: str = "en", score_script: str = "benchmark_fitting/score_voxpopuli_ref_errors.py",
                      hash_files: bool = True) -> dict:
    """Record predictions sha256 per arm and the exact argv of both tools. NOTHING is executed.

    The ref-disagreement tool reads one directory of per-model JSONL files (--preds) and needs
    the panel named with --panel (>=3 models, chosen independently of the arms under test).
    score_voxpopuli_ref_errors.py is the leaderboard scorer; its CLI is not pinned in the dive,
    so only `--predictions <file>` is assumed and must be confirmed at run time.
    """
    for a in arms:
        if a not in predictions_paths:
            raise KeyError(f"no predictions path for arm {a!r}")
    arms_plan = {}
    for a in arms:
        p = Path(predictions_paths[a])
        sha = sha256_file(p) if hash_files else None
        arms_plan[a] = {
            "predictions_path": str(p),
            "predictions_sha256": sha,
            "score_argv": ["python3", score_script, "--predictions", str(p)],
            "decisions_out": str(Path(out_dir) / f"{a}.decisions.jsonl"),
        }
    preds_dirs = sorted({str(Path(predictions_paths[a]).parent) for a in arms})
    argv = ["benchmark-optimization", "ref-disagreement", "--preds", preds_dirs[0] if len(preds_dirs) == 1 else "<PREDS_DIR>",
            "--language", language, "--panel", ",".join(panel_models), "--out", str(Path(out_dir) / "ref_disagreement.json")]
    return {
        "panel_commit": panel_commit,
        "tool_commit": BENCHMARK_OPTIMIZATION_COMMIT,
        "language": language,
        "executed": False,
        "ref_disagreement_argv": argv,
        "needs_single_preds_dir": len(preds_dirs) != 1,
        "arms": arms_plan,
    }


def screen_record(arm: str, items: Sequence[Mapping[str, Any]], *, model_revision: str, panel: str,
                  predictions_sha256: str, language: str = "en", stratum: str = "public-anchor",
                  probe: str = "ref_disagreement", control: bool = False) -> dict:
    rec = {
        "arm": arm, "model_revision": model_revision, "probe": probe, "panel": panel,
        "predictions_sha256": predictions_sha256, "language": language, "stratum": stratum,
        "is_control_baseline": bool(control or arm == CONTROL_ARM),
        "metric_direction": METRIC_DIRECTION, "gates": False,
    }
    if language != "en":
        rec.update(status="unmeasured", accept_ref=None, accept_ref_ci=None, n_eligible=0,
                   n_unparseable=0, label=None)
        return rec
    r = accept_ref(items)
    rec.update(status="measured", accept_ref=r["accept_ref"], accept_ref_ci=r["accept_ref_ci"],
               n_eligible=r["n_eligible"], n_unparseable=r["n_unparseable"],
               n_accept_ref=r["n_accept_ref"], label=label_for(r["accept_ref"]))
    return rec
