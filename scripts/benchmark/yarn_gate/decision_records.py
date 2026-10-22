"""VB-YARN-AGG-EMIT: build ``epyc.yarn_gate_decision.v1`` records and hand them to the belief-kernel adapter.

The record schema is defined by the ROOT adapter (``scripts/vidya/adapters/yarn_gate_agg.py``, module docstring and
``record_problems``). This module is the producer side: pure functions that turn arm results into cell / verdict
records, plus ``emit`` which writes each record to a durable JSON file (its sha256 is the row attestation) and calls
``yarn_gate_agg.write_belief_measurements`` ONCE per aggregation pass.

Rules enforced here (the adapter re-checks them):
  * a cell is ONE zone: results are grouped by (suite, task, REAL-length bin, zone), never "mixed";
  * only measured prompt counts are used; a result without ``real_n`` / ``n_source`` is excluded WITH a reason;
  * only suites the schema knows (oolong, helmet, longbio) emit; others (ruler) are excluded with a reason;
  * ``comparable_to_published`` is always false;
  * a record the adapter rejects is excluded with its problems listed, never silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.benchmark.yarn_gate import agg_scoring, retokenize  # noqa: E402

DECISION_SCHEMA = "epyc.yarn_gate_decision.v1"
SUITES = ("oolong", "helmet", "longbio")
MEASURED_SOURCES = retokenize.MEASURED_SOURCES
PIN_WORKTREE = "/mnt/raid0/llm/worktrees/root-pbe-3c4ca599"
ROOT_VIEW = "/mnt/raid0/llm/views/epyc-root-main"
ADAPTER_REL = "scripts/vidya/adapters/yarn_gate_agg.py"
PRODUCER = "yarn-gate-runner"
RECORD_DIR = "belief_records"
GATE_ARMS = {"A0": "A0", "A1": "A1"}
COMPARISON = "A0_vs_A1"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def prompt_n_stats(ns: Sequence[int], source: str) -> dict:
    return {"source": source, "count": len(ns), "min": min(ns), "median": int(statistics.median_low(ns)), "max": max(ns)}


def source_pins(suite: str, pins: Mapping[str, Any] | None) -> dict | None:
    """Map an item's ``pins`` onto the schema's per-suite ``source_pins``; None when a pin is not recorded."""
    p = pins or {}
    if suite == "oolong":
        if not p.get("dataset") or not p.get("vendored_sha256"):
            return None
        return {"source_revision": p["dataset"], "vendored_sha256": p["vendored_sha256"]}
    if suite == "helmet":
        cfg = p.get("config") or {}
        if not p.get("helmet_sha") or not cfg.get("path") or not cfg.get("sha256"):
            return None
        return {"helmet_pin": p["helmet_sha"], "config_file": cfg["path"], "config_sha256": cfg["sha256"],
                "max_test_samples": p.get("max_test_samples", cfg.get("max_test_samples"))}
    if suite == "longbio":
        return {"generator_pin": p["longbio_sha"]} if p.get("longbio_sha") else None
    return None


def _group_key(r: Mapping[str, Any]) -> tuple:
    return (r.get("suite"), r.get("task"), r.get("bin"), r.get("zone"))


def cell_name(d: Mapping[str, Any]) -> str:
    return (f"{d['comparison']}.{d['arm_role']}.{d['suite']}.{d['task']}.{d['bin']}.{d['zone']}.{d['metric']}.json"
            .replace("/", "_"))


def _usable(r: Mapping[str, Any]) -> tuple[bool, str]:
    if r.get("status") != "ok":
        return False, "status != ok"
    if r.get("suite") not in SUITES:
        return False, f"suite {r.get('suite')!r} has no slot in epyc.yarn_gate_decision.v1"
    if not isinstance(r.get("real_n"), int) or r.get("n_source") not in MEASURED_SOURCES or not r.get("bin") or not r.get("zone"):
        return False, "no measured real prompt count / bin / zone"
    return True, ""


def collect(results: Mapping[str, Mapping[str, Mapping[str, Any]]], items: Mapping[str, Mapping[str, Any]], *,
            model_pin: str, mock: bool = False, gate_item_ids: Iterable[str] | None = None
            ) -> tuple[list[dict], dict[tuple, list], list[dict]]:
    """-> (cell decisions, paired groups {(suite,task,bin,zone): [(item_id, r0, r1)]}, excluded-with-reason).

    ``results``: {arm: {item_id: result}}; ``items``: {item_id: frozen plan item}. Pairs are native on BOTH arms and,
    when ``gate_item_ids`` is given, restricted to those ids."""
    excluded: list[dict] = []
    cells: list[dict] = []
    for arm in ("A0", "A1"):
        groups: dict[tuple, list] = {}
        for iid, r in sorted(results.get(arm, {}).items()):
            ok, why = _usable(r)
            if not ok:
                excluded.append({"arm": arm, "item_id": iid, "reason": why})
                continue
            groups.setdefault(_group_key(r), []).append(r)
        for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
            suite, task, b, zone = key
            sources = {r["n_source"] for r in rs}
            if len(sources) != 1:
                excluded.append({"arm": arm, "group": list(key), "reason": f"mixed prompt_n sources {sorted(sources)}"})
                continue
            pins = {json.dumps(source_pins(suite, (items.get(r["item_id"]) or {}).get("pins")), sort_keys=True) for r in rs}
            if len(pins) != 1 or pins == {"null"}:
                excluded.append({"arm": arm, "group": list(key), "reason": "source pins absent or not uniform in the group"})
                continue
            ns_ = [r["real_n"] for r in rs]
            noms = [r["nominal_len"] for r in rs if isinstance(r.get("nominal_len"), int)]
            base = {"schema": DECISION_SCHEMA, "record_kind": "cell", "mock": mock, "comparison": COMPARISON,
                    "arm_role": arm, "suite": suite, "task": task,
                    "nominal_length": int(statistics.median_low(noms)) if noms else 0,
                    "model_pin": model_pin, "prompt_n": prompt_n_stats(ns_, next(iter(sources))), "bin": b, "zone": zone,
                    "source_pins": json.loads(next(iter(pins))), "comparable_to_published": False, "judged": False,
                    "n_units": len({r.get("context_id") for r in rs}) if suite == "oolong" else len(rs),
                    "unit": "contexts" if suite == "oolong" else "items", "n_unjudged": 0}
            cells.append({**base, "metric": "mean_score", "value": sum(r["score"] for r in rs) / len(rs)})
            cells.append({**base, "metric": "no_answer_rate",
                          "value": sum(bool(r.get("no_answer")) for r in rs) / len(rs)})
    gate = set(gate_item_ids) if gate_item_ids is not None else None
    pairs: dict[tuple, list] = {}
    for iid, r0 in sorted(results.get("A0", {}).items()):
        r1 = results.get("A1", {}).get(iid)
        if (gate is not None and iid not in gate) or not r1 or not (_usable(r0)[0] and _usable(r1)[0]):
            continue
        if r0["zone"] != "native" or r1["zone"] != "native":
            continue
        if _group_key(r0) != _group_key(r1):
            excluded.append({"item_id": iid, "reason": f"pair lands in different bin/zone A0={_group_key(r0)} A1={_group_key(r1)}"})
            continue
        pairs.setdefault(_group_key(r0), []).append((iid, r0, r1))
    return cells, pairs, excluded


def _write_json(path: Path, obj: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n").encode()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return _sha(data)


def locate_root() -> Path | None:
    """EPYC_ROOT, else the shared view, else the pinned worktree: the first that carries the adapter."""
    for c in (os.environ.get("EPYC_ROOT"), ROOT_VIEW, PIN_WORKTREE):
        if c and (Path(c) / ADAPTER_REL).is_file() and "def write_belief_measurements" in (Path(c) / ADAPTER_REL).read_text():
            return Path(c)
    return None


def load_adapter(root: Path | None = None):
    root = root or locate_root()
    if root is None:
        raise RuntimeError("no epyc-root checkout with scripts/vidya/adapters/yarn_gate_agg.py (EPYC_ROOT unset)")
    p = str(Path(root) / "scripts" / "vidya")
    if p not in sys.path:
        sys.path.insert(0, p)
    import importlib
    return importlib.import_module("adapters.yarn_gate_agg")


def emit(run_dir: Path, *, run_id: str, cells: Sequence[Mapping[str, Any]], pairs: Mapping[tuple, list],
         rule_id: str, alpha: float, final: bool, excluded: list[dict] | None = None, mock: bool = False,
         fixture: bool = False, root: Path | None = None, extra_records: Sequence[Mapping[str, Any]] = (),
         protocol_id: str | None = None) -> dict:
    """Write each record to ``run_dir/belief_records/`` and call ``write_belief_measurements`` once. INTERIM -> no call.
    Returns a status dict (never raises for adapter-side rejection of one record: that record is listed in
    ``excluded`` with the adapter's problems)."""
    excluded = list(excluded or [])
    if not final:
        return {"status": "skipped", "reason": "INTERIM decision: only FINAL decisions project", "excluded": excluded}
    adapter = load_adapter(root)
    rdir = Path(run_dir) / RECORD_DIR
    rows: list[dict] = []
    digest: dict[str, str] = {}

    def add(dec: Mapping[str, Any], path: Path, sha: str) -> None:
        bad = adapter.record_problems(dec, where=path.name)
        if bad:
            excluded.append({"record": path.name, "reason": bad})
        else:
            rows.append({"record_path": str(path), "decision": dict(dec)})

    for d in list(cells) + list(extra_records):
        path = rdir / cell_name(d)
        sha = _write_json(path, d)
        digest[path.name] = sha
        add(d, path, sha)
    for key, ps in sorted(pairs.items(), key=lambda kv: str(kv[0])):
        suite, task, b, zone = key
        names = {a: cell_name({"comparison": COMPARISON, "arm_role": a, "suite": suite, "task": task, "bin": b,
                               "zone": zone, "metric": "mean_score"}) for a in ("A0", "A1")}
        if any(n not in digest for n in names.values()):
            excluded.append({"group": list(key), "reason": "paired cell record missing or excluded: no verdict row"})
            continue
        s0 = {iid: r0["score"] for iid, r0, _ in ps}
        s1 = {iid: r1["score"] for iid, _, r1 in ps}
        mc = agg_scoring.net_paired_mcnemar(s0, s1, alpha=alpha)
        if not mc["n_pairs"]:
            continue
        ref = next(c for c in cells if c["arm_role"] == "A1" and (c["suite"], c["task"], c["bin"], c["zone"]) == key
                   and c["metric"] == "mean_score")
        pn = [r0["real_n"] for _, r0, _ in ps]
        ver = {"schema": DECISION_SCHEMA, "record_kind": "verdict", "mock": mock, "comparison": COMPARISON,
               "arm_role": "A1_vs_A0", "suite": suite, "task": task, "nominal_length": ref["nominal_length"],
               "model_pin": ref["model_pin"], "prompt_n": prompt_n_stats(pn, ps[0][1]["n_source"]), "bin": b,
               "zone": zone, "source_pins": ref["source_pins"], "comparable_to_published": False, "judged": False,
               "verdict": mc["verdict"], "rule_id": rule_id, "alpha": mc["alpha"], "p_one_sided": mc["p_one_sided"],
               "b_a0right_a1wrong": mc["b_a0right_a1wrong"], "c_a0wrong_a1right": mc["c_a0wrong_a1right"],
               "n_pairs": mc["n_pairs"], "mean_score_diff_a1_minus_a0": mc["mean_score_diff_a1_minus_a0"],
               "a0_record_sha256": digest[names["A0"]], "a1_record_sha256": digest[names["A1"]]}
        path = rdir / f"{COMPARISON}.A1_vs_A0.{suite}.{task}.{b}.{zone}.verdict.json".replace("/", "_")
        sha = _write_json(path, ver)
        add(ver, path, sha)
    if not rows:
        return {"status": "no-records", "excluded": excluded}
    sidecar = adapter.write_belief_measurements(run_dir, run_id=run_id, producer=PRODUCER, records=rows,
                                                protocol_id=protocol_id, fixture=fixture)
    return {"status": "written", "sidecar": str(sidecar), "n_records": len(rows), "excluded": excluded}
