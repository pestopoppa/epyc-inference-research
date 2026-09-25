#!/usr/bin/env python3
"""Read-only Vidya belief context for the AutoKernel planner, and its receipt.

WHAT IT ANSWERS. "Which ingested, graded claims apply to THIS planner target?" -- for any source,
not one hypothesis family. The planner names its target (model file, quant, backend, device,
optionally context depth); this module returns the claims whose declared applicability scope
equals it, rendered as a bounded, neutral evidence block, plus a receipt of what was presented,
omitted or unavailable and why. Nothing applicable -> no block at all.

A READ-ONLY CONSUMER OF ROOT. Everything that decides warrant is ROOT's, loaded by file from
`EPYC_ROOT_REPO` (default /workspace), the way `actors._seat_capture` loads VB-AK-SEAT:

  * `ledger.Ledger.verify()/read_all()` -- an integrity failure presents nothing;
  * `fold.fold()` -- retraction, supersession, corrections, aliases;
  * `gate.evaluate()` under the declared `POLICY` -- the use-policy gate `cli.py query` applies,
    WITHOUT writing a `query_served` frame. Anything the gate does not ALLOW (conflicted,
    review-required, dirty, retracted, below floor) is omitted with the gate's reason;
  * the grade shown is the folded `Belief.pro`: what `claim_tuple.grade()` assigned at the write
    boundary, as it survives the fold. No grading rule lives here.

THE SCOPE GAP. `claim_tuple.to_frames()` persists metric, text, locator, source kind and grade --
not `ClaimTuple.extra`, where adapters keep model/backend/device. So no ledger frame carries a
structured applicability scope, and inventing one on read would claim warrant the write never
captured. Matching is therefore conservative: a source is eligible only when its PRODUCER
CONTRACT pins the whole scope for every row it can emit, declared once in `SOURCE_SCOPES` with the
`basis` that pins it. An undeclared source never matches. (Proposal to close the gap in the shared
projection API: /mnt/raid0/llm/tmp/vidya-planner-receipt-proposal-20260925.md section 4.)

MATCHING (exact, never nearest): GGUF basename byte-equal; quant equal after upper-casing;
backend equal; device equal; context depth equal only when BOTH sides declare one. The
measurement surface is rendered, not matched: no source declares it in the planner's vocabulary.

DECISION-READINESS. Per source, only the NEWEST COMPLETE run is presented: a run is complete when
its gate-ALLOWed claims cover exactly the source's declared key set. Incomplete, superseded,
gate-refused and target-mismatched evidence is omitted and named in the receipt.

Stdlib only and no relative imports: the planner runs this file as a child process
(`python3 belief_context.py --target-json ...`) so a slow ledger can be killed at the deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Callable, Iterable, Mapping

ROOT_REPO_ENV = "EPYC_ROOT_REPO"
DEFAULT_ROOT = Path("/workspace")
READER_TIMEOUT_S = 10.0
RECEIPT_SCHEMA = "epyc.vidya.planner_evidence_receipt.v1"
RECEIPT_LOG = "belief-receipts.jsonl"
#: Env key carrying the presented-evidence summary to the metrics row (the
#: `AK_ACTOR_SEAT_ARM` precedent: a second channel into `_record_metrics`, harmless to the child).
ENV_KEY = "AK_ACTOR_BELIEF_CONTEXT"
SECTION_HEADER = "## Belief-kernel evidence (Vidya ledger, read-only)"
RELIANCE_FIELD = "relies_on_claims"
DEFAULT_MAX_CLAIMS = 12
DEFAULT_MAX_BYTES = 10000
CLAIM_TEXT_CAP = 440
REASON_CAP = 200
OMITTED_CAP = 24
#: The declared use policy. Floor `Judged/Located`: the lowest grade at which a claim names where
#: it came from; an observation is shown AS an observation (its grade printed beside it).
POLICY = {"use": "planner-context", "floor": "Judged/Located", "standard": "DV"}
#: The only GPU on this host is the MI210 (the planner prompt already names gfx90a); the CPU
#: target is the EPYC 9655 host. A target is matched on these literal device names.
DEVICE_BY_BACKEND = {"gpu": "gfx90a", "cpu": "epyc-9655"}
_VIDYA_MODULES = ("canonical", "lattice", "frames", "ledger", "fold", "gate")


# ------------------------------------------------------------------------------ target


@dataclass(frozen=True)
class PlannerTarget:
    model_file: str
    quant: str
    backend: str
    device: str
    surface: str = ""
    context_tokens: int | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PlannerTarget":
        """`model_file`, or `model_path` reduced to its basename. An incomplete target raises:
        it is not a wildcard."""
        model = str(raw.get("model_file") or "").strip()
        if not model and raw.get("model_path"):
            model = PurePath(str(raw["model_path"])).name
        values = {"model_file": model, "quant": str(raw.get("quant") or "").strip(),
                  "backend": str(raw.get("backend") or "").strip().lower(),
                  "device": str(raw.get("device") or "").strip().lower()}
        missing = sorted(key for key, value in values.items() if not value)
        if missing:
            raise ValueError(f"target incomplete: missing {missing}")
        if values["backend"] not in DEVICE_BY_BACKEND:
            raise ValueError(f"backend must be cpu or gpu (got {values['backend']!r})")
        depth = raw.get("context_tokens")
        if depth is not None and (not isinstance(depth, int) or isinstance(depth, bool)
                                  or depth < 1):
            raise ValueError("context_tokens must be a positive integer when given")
        return cls(surface=str(raw.get("surface") or ""), context_tokens=depth, **values)


def target_from_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """The planner's target, from the `current_regime` block `run.py` already builds."""
    regime = context.get("current_regime") or {}
    model = regime.get("model") or {}
    backend = str(regime.get("backend") or "").lower()
    return {"model_path": str(model.get("path") or "") if isinstance(model, Mapping) else "",
            "quant": regime.get("quant") or "", "backend": backend,
            "device": DEVICE_BY_BACKEND.get(backend, ""),
            "surface": regime.get("measurement_surface") or ""}


# ------------------------------------------------------------------------------ scopes


@dataclass(frozen=True)
class SourceScope:
    """A source whose producer contract pins its whole applicability scope."""

    source_kind: str
    model_file: str
    quant: str
    backend: str
    device: str
    #: What pins this scope for EVERY row the source can emit. Required.
    basis: str
    #: Neutral statement of what the evidence is and is not; never advice.
    limits: str
    #: Locator shape the ADAPTER enforces; named groups `run`, `depth` and the key groups.
    locator_re: re.Pattern[str] | None = None
    key_groups: tuple[str, ...] = ()
    #: The complete per-run key set (tuples over `key_groups`); empty = no run semantics.
    expected_keys: frozenset[tuple[str, ...]] = frozenset()
    depth_tokens: Mapping[str, int] = field(default_factory=dict)

    def applies_to(self, target: PlannerTarget) -> bool:
        return (target.model_file == self.model_file
                and target.quant.upper() == self.quant.upper()
                and target.backend == self.backend and target.device == self.device)

    def parse(self, locator: str) -> tuple[str | None, tuple[str, ...] | None, int | None]:
        """(run, key, context_tokens) from the locator; Nones when the shape does not match."""
        if self.locator_re is None:
            return None, None, None
        match = self.locator_re.match(locator)
        if match is None:
            return None, None, None
        groups = match.groupdict()
        key = tuple(groups.get(name) or "" for name in self.key_groups) or None
        return groups.get("run"), key, self.depth_tokens.get(groups.get("depth") or "")


_KVQ_ARMS = ("A_f16_kv", "B_q8_0_kv", "C_q4_0_kv")
_KVQ_DEPTHS = ("d2k", "d32k")
_KVQ_METRICS = ("gpu_decode_tps", "gpu_prefill_tps")

SOURCE_SCOPES: dict[str, SourceScope] = {scope.source_kind: scope for scope in (
    SourceScope(
        source_kind="kv-quant-27b-v10-measurement",
        model_file="Qwen3.8-27B-Q8_0.gguf", quant="Q8_0", backend="gpu", device="gfx90a",
        basis=("producer scripts/benchmark/kv_quant_27b_v10_sweep.py refuses any model other "
               "than /mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf at its pinned byte length "
               "(model_identity_valid) and runs on MI210 device mi210_0; ROOT adapter "
               "kv_quant_27b_v10.py pins the producer bytes by sha256 and calls validate_row()"),
        limits=("llama-bench instrument, bursty duty cycle (not a serving rate); "
                "production-consolidated-v10 kernel; -fa on in every arm; K and V cache types "
                "equal per arm (f16, q8_0, q4_0), mixed K/V not measured; prefill depth is part "
                "of each claim's key"),
        locator_re=re.compile(r"^kvq:(?P<run>[^:|]+):(?P<arm>[^:|]+):(?P<depth>[^:|]+):"
                              r"(?P<metric>[^:|]+)(?:\||$)"),
        key_groups=("arm", "depth", "metric"),
        expected_keys=frozenset((a, d, m) for a in _KVQ_ARMS for d in _KVQ_DEPTHS
                                for m in _KVQ_METRICS),
        depth_tokens={"d2k": 2048, "d32k": 32768}),
)}


def applicable_scopes(target: PlannerTarget) -> dict[str, SourceScope]:
    return {kind: scope for kind, scope in SOURCE_SCOPES.items() if scope.applies_to(target)}


# ------------------------------------------------------------------------------ ROOT API


def root_repo(root: str | Path | None = None) -> Path:
    return Path(root or os.environ.get(ROOT_REPO_ENV) or DEFAULT_ROOT)


def load_root(root: str | Path | None = None) -> dict[str, Any]:
    """ROOT's ledger/fold/gate/lattice, loaded from `<root>/scripts/vidya` and refused unless
    every module actually came from there (a same-named module from elsewhere is not ROOT)."""
    vidya = (root_repo(root) / "scripts" / "vidya").resolve()
    for name in _VIDYA_MODULES:
        if not (vidya / f"{name}.py").is_file():
            raise FileNotFoundError(f"ROOT reader module missing: {vidya / name}.py")
    sys.path.insert(0, str(vidya))
    try:
        modules = {name: importlib.import_module(name) for name in _VIDYA_MODULES}
    finally:
        try:
            sys.path.remove(str(vidya))
        except ValueError:
            pass
    for name, module in modules.items():
        if Path(module.__file__).resolve() != vidya / f"{name}.py":
            raise ImportError(f"module {name!r} resolves to {module.__file__}, not ROOT {vidya}")
    gate = modules["gate"]
    policy = gate.UsePolicy(use=POLICY["use"], floor=modules["lattice"].parse_grade(POLICY["floor"]),
                            standard=POLICY["standard"])
    return {"Ledger": modules["ledger"].Ledger, "fold": modules["fold"].fold,
            "evaluate": gate.evaluate, "allow": gate.Outcome.ALLOW, "policy": policy,
            "vidya": vidya}


# ------------------------------------------------------------------------------ reading


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _cap(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    head = (limit * 2) // 3
    return f"{text[:head]} ... {text[-(limit - head - 5):]}"


def _omit(source_kind: str, reason: str, *, run: str | None = None,
          claim_id: str | None = None, detail: str = "") -> dict[str, Any]:
    return {"source_kind": source_kind, "run": run, "claim_id": claim_id, "reason": reason,
            "detail": _cap(detail, REASON_CAP) if detail else ""}


def _empty(status: str, reasons: Iterable[str], *, as_of: str, frontier: int | None = None,
           omitted: list[dict] | None = None, error: str | None = None) -> dict[str, Any]:
    omitted = omitted or []
    return {"status": status, "reasons": sorted(set(reasons)), "frontier": frontier,
            "as_of": as_of, "claim_ids": [], "claims": [], "runs": [],
            "omitted": omitted[:OMITTED_CAP], "omitted_count": len(omitted),
            "section": "", "section_sha256": "", "section_bytes": 0, "error": error}


def evaluate_frames(frames: list[Mapping[str, Any]], target: PlannerTarget, api: Mapping[str, Any],
                    *, as_of: str, max_claims: int = DEFAULT_MAX_CLAIMS,
                    max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    """The evidence half of a receipt, from ledger frames already verified."""
    eligible = applicable_scopes(target)
    if not eligible:
        return _empty("omitted", ["inapplicable_target"], as_of=as_of)
    cutoff = _ts(as_of)
    visible = [f for f in frames if not (f.get("pubinfo") or {}).get("created_at")
               or _ts(f["pubinfo"]["created_at"]) <= cutoff]
    result = api["fold"](visible, as_of=as_of)
    sources: dict[str, Mapping[str, Any]] = {}
    claims: dict[str, Mapping[str, Any]] = {}
    support: dict[str, Mapping[str, Any]] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    for frame in visible:
        kind, assertion = str(frame.get("frame_type", "")), frame.get("assertion") or {}
        if kind.endswith("source_observed/v1") and assertion.get("source_kind") in eligible:
            sources[str(assertion.get("source_id", ""))] = assertion
        elif kind.endswith("claim_proposed/v1"):
            claims[str(assertion.get("claim_id", ""))] = assertion
        elif kind.endswith("evidence_supports_claim/v1"):
            cid = str(assertion.get("claim_id", ""))
            support[cid] = assertion
            reasons[cid] = tuple(str(r) for r in
                                 (frame.get("provenance") or {}).get("grade_reasons") or ())

    omitted: list[dict[str, Any]] = []
    runs: dict[tuple[str, str | None], list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for raw_id, claim in claims.items():
        source = sources.get(str(claim.get("source_id", "")))
        if source is None:
            continue
        scope = eligible[str(source["source_kind"])]
        claim_id = result.alias_map.get(raw_id, raw_id)
        if claim_id in seen:
            continue
        seen.add(claim_id)
        locator = str(source.get("locator", ""))
        run, key, depth = scope.parse(locator)
        if scope.expected_keys and (run is None or key not in scope.expected_keys):
            omitted.append(_omit(scope.source_kind, "incomplete_run", run=run, claim_id=claim_id,
                                 detail=f"locator {locator.split('|', 1)[0]!r} outside the "
                                        "declared run key set"))
            continue
        verdict = api["evaluate"](claim_id, result, api["policy"])
        if verdict.outcome != api["allow"]:
            omitted.append(_omit(scope.source_kind, f"gate_refused:{verdict.outcome}", run=run,
                                 claim_id=claim_id,
                                 detail=(verdict.reasons or [""])[0]))
            continue
        row = support.get(raw_id, {})
        runs.setdefault((scope.source_kind, run), []).append({
            "claim_id": claim_id, "source_kind": scope.source_kind, "run": run, "key": key,
            "context_tokens": depth, "grade": str(result.beliefs[claim_id].pro),
            "locator": locator.split("|", 1)[0], "revision": str(source.get("revision_observed", "")),
            "protocol_id": str(row.get("protocol_id") or ""),
            "reps": row.get("reps") if isinstance(row.get("reps"), int) else None,
            "text": str(claim.get("display_text", "")), "grade_reasons": reasons.get(raw_id, ())})

    if not runs and not omitted:
        return _empty("omitted", ["no_ingested_claims"], as_of=as_of, frontier=result.frontier)

    run_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for kind, scope in sorted(eligible.items()):
        groups = {run: rows for (source_kind, run), rows in runs.items() if source_kind == kind}
        refused_runs = {o["run"] for o in omitted if o["source_kind"] == kind and o["run"]}
        if not scope.expected_keys:
            for rows in groups.values():
                candidates.extend(rows)
            continue
        complete, incomplete = [], []
        for run in set(groups) | refused_runs:
            rows = groups.get(run, [])
            keys = [row["key"] for row in rows]
            if len(keys) == len(set(keys)) and set(keys) == scope.expected_keys:
                complete.append((max(row["revision"] for row in rows), run, rows))
            else:
                incomplete.append((run, len(set(keys))))
        for run, have in sorted(incomplete, key=lambda item: str(item[0])):
            detail = (f"{have} of {len(scope.expected_keys)} declared keys pass the gate")
            omitted.append(_omit(kind, "incomplete_run", run=run, detail=detail))
            run_rows.append({"source_kind": kind, "run": run, "status": "omitted",
                             "reason": "incomplete_run", "claim_ids": []})
        complete.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
        for index, (_, run, rows) in enumerate(complete):
            if index == 0:
                candidates.extend(rows)
                continue
            omitted.append(_omit(kind, "superseded_run", run=run,
                                 detail=f"newer complete run {complete[0][1]}"))
            run_rows.append({"source_kind": kind, "run": run, "status": "omitted",
                             "reason": "superseded_run",
                             "claim_ids": sorted(row["claim_id"] for row in rows)})

    if target.context_tokens is not None:
        kept = []
        for row in candidates:
            if row["context_tokens"] is not None and row["context_tokens"] != target.context_tokens:
                omitted.append(_omit(row["source_kind"], "inapplicable_target", run=row["run"],
                                     claim_id=row["claim_id"],
                                     detail=f"context {row['context_tokens']} != target "
                                            f"{target.context_tokens}"))
            else:
                kept.append(row)
        candidates = kept
    candidates.sort(key=lambda row: (row["source_kind"], row["locator"]))
    section, shown = render(target, candidates, frontier=result.frontier, as_of=as_of,
                            max_claims=max_claims, max_bytes=max_bytes)
    shown_ids = {row["claim_id"] for row in shown}
    for row in candidates:
        if row["claim_id"] not in shown_ids:
            omitted.append(_omit(row["source_kind"], "bounded", run=row["run"],
                                 claim_id=row["claim_id"],
                                 detail=f"cap {max_claims} claims / {max_bytes} bytes"))
    presented_runs: dict[tuple[str, str | None], list[str]] = {}
    for row in shown:
        presented_runs.setdefault((row["source_kind"], row["run"]), []).append(row["claim_id"])
    run_rows = [{"source_kind": kind, "run": run, "status": "presented", "reason": "",
                 "claim_ids": sorted(ids)} for (kind, run), ids in sorted(
                     presented_runs.items(), key=lambda item: (item[0][0], str(item[0][1])))] + run_rows
    codes = {o["reason"] for o in omitted}
    status = "presented" if shown else "omitted"
    out = {"status": status, "reasons": sorted(codes), "frontier": result.frontier,
           "as_of": as_of, "claim_ids": [row["claim_id"] for row in shown],
           "claims": [{k: row[k] for k in ("claim_id", "source_kind", "run", "grade", "locator")}
                      for row in shown],
           "runs": run_rows, "omitted": omitted[:OMITTED_CAP], "omitted_count": len(omitted),
           "section": section,
           "section_sha256": hashlib.sha256(section.encode()).hexdigest() if section else "",
           "section_bytes": len(section.encode()), "error": None}
    return out


def render(target: PlannerTarget, rows: list[dict[str, Any]], *, frontier: int | None,
           as_of: str, max_claims: int = DEFAULT_MAX_CLAIMS,
           max_bytes: int = DEFAULT_MAX_BYTES) -> tuple[str, list[dict[str, Any]]]:
    """Bounded, neutral block and the rows it carries; ("", []) when nothing fits."""
    if not rows or max_claims < 1:
        return "", []
    depth = f", context {target.context_tokens} tokens" if target.context_tokens else ""
    head = [SECTION_HEADER,
            f"Target: model {target.model_file}, quant {target.quant}, backend {target.backend}, "
            f"device {target.device}{depth}.",
            "Ingested claims whose declared applicability scope equals this target. Grade is the "
            f"folded claim_tuple.grade() Q/T at ledger frontier {frontier}, as of {as_of}."]
    tail = [f"Optional reply field {RELIANCE_FIELD}: a list of the claim ids above that the "
            "hypothesis depends on; omit it when it depends on none."]
    body: list[str] = []
    shown: list[dict[str, Any]] = []
    described: set[tuple[str, str | None]] = set()

    def size(lines: list[str]) -> int:
        return len("\n".join(head + lines + tail).encode("utf-8")) + 100   # + the count line

    def lines_for(row: dict[str, Any], first: bool) -> list[str]:
        lines: list[str] = []
        if first:
            scope = SOURCE_SCOPES[row["source_kind"]]
            run = f", run {row['run']}" if row["run"] else ""
            lines.append(f"Source {scope.source_kind}{run}: scope {scope.model_file} / "
                         f"{scope.quant} / {scope.backend} / {scope.device}. Limits: "
                         f"{scope.limits}.")
        facts = [f"protocol {row['protocol_id'] or 'none'}",
                 f"n={row['reps']}" if row["reps"] is not None else "n unrecorded"]
        if row["context_tokens"] is not None:
            facts.insert(0, f"context {row['context_tokens']}")
        if row["revision"]:
            facts.append(row["revision"])
        lines.append(f"- {row['claim_id']} [{row['grade']}] {row['locator']} "
                     f"({'; '.join(facts)}): {_cap(row['text'], CLAIM_TEXT_CAP)}")
        if row["grade_reasons"]:
            lines.append(f"  grade notes: {_cap(row['grade_reasons'][0], REASON_CAP)}")
        return lines

    # Packing unit: a whole run for a source with a declared run key set (a complete run is
    # shown whole or not at all -- a truncated matrix would read as an incomplete one), else
    # one claim. A unit that does not fit is skipped; a later, smaller unit may still fit.
    units: list[list[dict[str, Any]]] = []
    for row in rows:
        if (SOURCE_SCOPES[row["source_kind"]].expected_keys and units
                and (units[-1][0]["source_kind"], units[-1][0]["run"])
                == (row["source_kind"], row["run"])):
            units[-1].append(row)
        else:
            units.append([row])
    for unit in units:
        if len(shown) + len(unit) > max_claims:
            continue
        lines: list[str] = []
        for index, row in enumerate(unit):
            lines.extend(lines_for(row, index == 0 and (row["source_kind"], row["run"])
                                   not in described))
        if size(body + lines) > max_bytes:
            continue
        body.extend(lines)
        described.update((row["source_kind"], row["run"]) for row in unit)
        shown.extend(unit)
    if not shown:
        return "", []
    count = (f"Shown {len(shown)} of {len(rows)} decision-ready claim(s); bounded at "
             f"{max_claims} claims / {max_bytes} bytes.")
    return "\n".join(head + body + [count] + tail), shown


def read(target: PlannerTarget | Mapping[str, Any], *, root: str | Path | None = None,
         ledger_path: str | Path | None = None, as_of: str | None = None,
         max_claims: int = DEFAULT_MAX_CLAIMS, max_bytes: int = DEFAULT_MAX_BYTES,
         api: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The evidence dict for one target. Never raises."""
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    try:
        if not isinstance(target, PlannerTarget):
            target = PlannerTarget.from_mapping(target)
    except ValueError as exc:
        return _empty("omitted", ["inapplicable_target"], as_of=as_of, error=str(exc))
    if not applicable_scopes(target):
        return _empty("omitted", ["inapplicable_target"], as_of=as_of)
    try:
        api = api or load_root(root)
    except Exception as exc:  # noqa: BLE001
        return _empty("unavailable", ["root_missing"], as_of=as_of,
                      error=f"{type(exc).__name__}: {exc}"[:300])
    path = Path(ledger_path) if ledger_path else root_repo(root) / ".vidya" / "ledger.jsonl"
    if not path.is_file():
        return _empty("unavailable", ["ledger_missing"], as_of=as_of, error=str(path))
    try:
        ledger = api["Ledger"](path)
        errors = ledger.verify()
        if errors:
            return _empty("unavailable", ["ledger_integrity"], as_of=as_of,
                          error=str(errors[0])[:300])
        frames = [record.frame for record in ledger.read_all()]
        return evaluate_frames(frames, target, api, as_of=as_of, max_claims=max_claims,
                               max_bytes=max_bytes)
    except Exception as exc:  # noqa: BLE001 -- a planner context is never a reason to fail
        return _empty("unavailable", ["reader_error"], as_of=as_of,
                      error=f"{type(exc).__name__}: {exc}"[:300])


# ------------------------------------------------------------------------------ planner side


def planner_evidence(context: Mapping[str, Any], *, root: str | Path | None = None,
                     timeout_s: float = READER_TIMEOUT_S,
                     runner: Callable[..., Any] = subprocess.run) -> dict[str, Any]:
    """What `AgentPlanner.propose` calls. Never raises and never exceeds `timeout_s`.

    A target no declared scope applies to is answered here without a child process (the DS41
    CPU target never opens the ledger). Otherwise the reader runs as a child so a slow ledger
    is killed at the deadline; any failure is `unavailable` with a reason, never an exception."""
    started = time.monotonic()
    raw = target_from_context(context)
    as_of = datetime.now(timezone.utc).isoformat()
    try:
        target = PlannerTarget.from_mapping(raw)
    except ValueError as exc:
        evidence = _empty("omitted", ["inapplicable_target"], as_of=as_of, error=str(exc))
    else:
        if not applicable_scopes(target):
            evidence = _empty("omitted", ["inapplicable_target"], as_of=as_of)
        else:
            argv = [sys.executable, str(Path(__file__).resolve()), "--target-json",
                    json.dumps(asdict(target), sort_keys=True), "--root", str(root_repo(root)),
                    "--as-of", as_of]
            try:
                done = runner(argv, capture_output=True, text=True, timeout=timeout_s,
                              check=False)
                evidence = json.loads(done.stdout)
                if not isinstance(evidence, dict) or "status" not in evidence:
                    raise ValueError("reader printed no evidence object")
            except subprocess.TimeoutExpired:
                evidence = _empty("unavailable", ["reader_timeout"], as_of=as_of,
                                  error=f"reader exceeded {timeout_s}s")
            except Exception as exc:  # noqa: BLE001
                evidence = _empty("unavailable", ["reader_error"], as_of=as_of,
                                  error=f"{type(exc).__name__}: {exc}"[:300])
    evidence["target"] = raw
    evidence["elapsed_s"] = round(time.monotonic() - started, 3)
    return evidence


def reliance(declared: Any, presented: Iterable[str]) -> dict[str, Any]:
    """Shown is not relied on: only an explicit reply declaration counts, and only for ids that
    were presented. Anything else is recorded as rejected, never raised."""
    items = declared if isinstance(declared, list) else ([] if declared is None else [declared])
    presented = set(presented)
    accepted, rejected = [], []
    for item in items:
        if isinstance(item, str) and item in presented:
            if item not in accepted:
                accepted.append(item)
        else:
            rejected.append({"claim_id": item if isinstance(item, str) else repr(item)[:80],
                             "reason": "not_presented"})
    return {"declared": [i if isinstance(i, str) else repr(i)[:80] for i in items],
            "accepted": accepted, "rejected": rejected}


def metrics_summary(evidence: Mapping[str, Any]) -> str:
    """The presented half, as the env value `_record_metrics` folds into the metrics row."""
    return json.dumps({"status": evidence.get("status"), "frontier": evidence.get("frontier"),
                       "claim_ids": list(evidence.get("claim_ids") or []),
                       "runs": [{"source_kind": r.get("source_kind"), "run": r.get("run"),
                                 "status": r.get("status")} for r in evidence.get("runs") or []],
                       "reasons": list(evidence.get("reasons") or [])}, sort_keys=True)


def _module_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return ""


def seal_receipt(evidence: Mapping[str, Any] | None, *, knob: str, prompt: str,
                 workspace: str | Path, seat_arm: str | None, outcome: Mapping[str, Any],
                 relied: Mapping[str, Any] | None = None, root: str | Path | None = None,
                 consumer: str = "autokernel.planner") -> dict[str, Any]:
    evidence = dict(evidence or _empty("omitted", ["knob_off"],
                                       as_of=datetime.now(timezone.utc).isoformat()))
    body = {
        "schema": RECEIPT_SCHEMA, "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "consumer": consumer,
        "call": {"prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                 "workspace": str(workspace), "seat_arm": seat_arm},
        "knob": knob, "target": evidence.get("target"),
        "reader": {"module": "scripts/kernel_rnd/autokernel/loop/belief_context.py",
                   "module_sha256": _module_sha256(), "root_repo": str(root_repo(root)),
                   "policy": dict(POLICY), "elapsed_s": evidence.get("elapsed_s"),
                   "error": evidence.get("error")},
        "run": list(evidence.get("runs") or []), "frontier": evidence.get("frontier"),
        "as_of": evidence.get("as_of"), "claim_ids": list(evidence.get("claim_ids") or []),
        "evidence_status": evidence.get("status"),
        "evidence_reasons": list(evidence.get("reasons") or []),
        "omitted": list(evidence.get("omitted") or []),
        "omitted_count": int(evidence.get("omitted_count") or 0),
        "section_sha256": evidence.get("section_sha256") or "",
        "section_bytes": int(evidence.get("section_bytes") or 0),
        "outcome": dict(outcome),
        "reliance": dict(relied or reliance(None, ())),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return {**body, "receipt_id": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()}


def write_receipt(log_dir: str | Path, receipt: Mapping[str, Any]) -> None:
    """Append one receipt line. Evidence, never a reason to fail the planner call."""
    try:
        target = Path(log_dir)
        target.mkdir(parents=True, exist_ok=True)
        with open(target / RECEIPT_LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, sort_keys=True, default=str) + "\n")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-json", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-claims", type=int, default=DEFAULT_MAX_CLAIMS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.target_json)
        if not isinstance(raw, dict):
            raise ValueError("target must be a JSON object")
    except ValueError as exc:
        evidence = _empty("omitted", ["inapplicable_target"],
                          as_of=args.as_of or datetime.now(timezone.utc).isoformat(),
                          error=f"bad target: {exc}")
    else:
        evidence = read(raw, root=args.root, ledger_path=args.ledger, as_of=args.as_of,
                        max_claims=args.max_claims, max_bytes=args.max_bytes)
    print(json.dumps(evidence, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
