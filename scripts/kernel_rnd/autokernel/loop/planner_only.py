#!/usr/bin/env python3
"""`run.py --planner-only`: one real planner turn and, if it proposes, the authoring step.

Answers one question in minutes and with no whole-host window: is planning guided by belief
knowledge? It uses the real belief read, a real source tree and an already-captured profile,
then stops before build or measurement. It takes no floor, GPU device lock or CPU region lock
and starts no server. The source tree is never edited: the actors work in a throwaway clone.

The report records and does not judge; whether the turn was guided is decided by reading it.
An absent profile or source tree is refused: the empty 11-08 turn abstained meaninglessly.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..controller import workload_contract
from . import actors, belief_context, loop

REPORT_NAME = "planner-only-report.json"
SCHEMA = "epyc.autokernel.planner_only_report.v1"
PROFILE_KEYS = ("kernel_hotspots", "gpu_serving_profile")
#: Where the GPU route's sources live: the tree the planner is told to target.
GPU_SOURCE_DIR = "ggml/src/ggml-cuda"


class Refused(RuntimeError):
    """A required input is absent: no turn is run."""


class _RecordingPlanner(actors.AgentPlanner):
    """The real planner, keeping the ready answer `propose` obtains for the report."""

    last_ready: belief_context.ReadyAnswer | None = None

    def _belief_ready(self, context):
        self.last_ready = super()._belief_ready(context)
        return self.last_ready


def add_arguments(parser) -> None:
    parser.add_argument("--planner-only", action="store_true",
                        help="one real planner turn and authoring step, then stop before build; "
                             "needs --out, --model, --worktree and a profile; takes no lock")
    parser.add_argument("--profile", type=Path,
                        help="--planner-only profile JSON (status file with hotspots, context "
                             "object or rows); default <store>/loop-status.json")


def load_profile(path: Path) -> dict[str, Any]:
    """Planner context keys from a status file (`hotspots`), a context object or a bare row list."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refused(f"no usable profile at {path}: {exc}") from exc
    if isinstance(raw, list):
        raw = {"kernel_hotspots": raw}
    if not isinstance(raw, Mapping):
        raise Refused(f"profile {path} is neither an object nor a list of rows")
    found = dict(raw)
    if not found.get("kernel_hotspots") and found.get("hotspots"):
        found["kernel_hotspots"] = found["hotspots"]
    profile = {key: found[key] for key in PROFILE_KEYS if found.get(key)}
    if not profile:
        raise Refused(f"profile {path} has no hotspot rows: pass --profile with a captured "
                      "profile, or a --store whose loop-status.json holds `hotspots`")
    return profile


def _git(tree: Path, *argv: str) -> str:
    return subprocess.run(["git", "-C", str(tree), *argv], check=True, capture_output=True,
                          text=True).stdout.strip()


def require_source(tree: Path) -> str:
    """The commit of a real GPU source tree, or `Refused`."""
    if not (tree / GPU_SOURCE_DIR).is_dir() or not any((tree / GPU_SOURCE_DIR).iterdir()):
        raise Refused(f"source tree {tree} has no {GPU_SOURCE_DIR}/ sources")
    try:
        return _git(tree, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Refused(f"source tree {tree} is not a git checkout: {exc}") from exc


def claim_texts(section: str, claim_ids: list[str]) -> dict[str, str]:
    """Each claim's text as the planner saw it: its head line and indented continuation."""
    texts: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in section.splitlines():
        if line.startswith("- "):
            ident = line[2:].split(maxsplit=1)[0] if line[2:].strip() else ""
            current = texts.setdefault(ident, []) if ident in claim_ids else None
            if current is not None:
                current.append(line[2:])
        elif current is not None and line.startswith("  "):
            current.append(line.strip())
        else:
            current = None
    return {claim: "\n".join(texts[claim]) if claim in texts else "" for claim in claim_ids}


def _last_receipt(log_dir: Path) -> dict[str, Any]:
    try:
        lines = (log_dir / belief_context.RECEIPT_LOG).read_text(encoding="utf-8").splitlines()
        return json.loads(lines[-1]) if lines else {}
    except (OSError, ValueError):
        return {}


def _plain(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def _report(planner, result: Any, *, status: str, source_commit: str, profile_ref: str,
            author: Any, detail: str = "") -> dict[str, Any]:
    receipt = _last_receipt(Path(planner.workspace).parent / actors.ACTOR_REPLY_DIR)
    ready = getattr(planner, "last_ready", None)
    evidence = ready.evidence if ready is not None else {}
    state = ready.ledger_state if ready is not None else {}
    declared = (receipt.get("reliance") or {}).get("declared") or []
    relied = (receipt.get("reliance") or {}).get("accepted") or []
    texts = claim_texts(str(evidence.get("section") or ""), list(dict.fromkeys(declared + relied)))
    return {
        "schema": SCHEMA, "status": status, "detail": detail,
        "hypothesis": getattr(result, "statement", None),
        "mechanism_id": getattr(result, "mechanism_id", None),
        "target_surface": getattr(result, "target_surface", None),
        "target_symbol": getattr(result, "target_symbol", None),
        "abstain_reason": getattr(result, "reason", None),
        "relies_on_claims": relied,
        "reliance": receipt.get("reliance"),
        "cited_claims": [{"claim_id": claim, "text": text} for claim, text in texts.items()],
        "presented_claim_ids": list(evidence.get("claim_ids") or []),
        # The reader's read log carries no id per record; this digest names the ledger state read.
        "read_record_id": receipt.get("ledger_state_digest"),
        "read_record": {"schema": "epyc.vidya.read_record.v1",
                        "ledger_path": ready.answer.get("ledger_path") if ready else None,
                        "head_hash": state.get("head_hash"),
                        "frame_count": state.get("frame_count"),
                        "read_log_error": ready.answer.get("read_log_error") if ready else None},
        "receipt_id": receipt.get("receipt_id"),
        "source_commit": source_commit,
        "profile": profile_ref,
        "author": _plain(author),
        "raw_result": _plain(result),
    }


def run(args, *, seat_common: Mapping[str, Any],
        make_planner: Callable[..., Any] | None = None) -> int:
    """Exit 0 on a recorded turn (propose or abstain), 1 on an actor or belief failure, 2 on a
    refusal of its inputs."""
    try:
        if args.out is None:
            raise Refused("--planner-only needs --out DIR for the report and throwaway clone")
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            raise Refused(f"--out {out} is not empty")
        if args.model is None or not Path(args.model).is_file():
            raise Refused(f"--model {args.model} does not exist")
        source_commit = require_source(Path(args.worktree))
        profile_path = Path(args.profile or Path(args.store) / "loop-status.json")
        profile = load_profile(profile_path)
    except Refused as exc:
        print(f"planner-only REFUSED: {exc}", file=sys.stderr)
        return 2
    workspace = out / "ws"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--shared", str(args.worktree), str(workspace)],
                   check=True)
    subprocess.run(["git", "-C", str(workspace), "checkout", "-q", "--detach", source_commit],
                   check=True)
    census = workload_contract.read_split_census(args.model)
    backend = actors.backend_for(args.planner_model, args.planner_effort)
    context = {
        "program": loop.PROGRAM.read_text(encoding="utf-8"),
        "actor_provenance": {"planner": backend.describe()},
        "actor_read_roots": [str(out.resolve())],
        **profile,
        "current_regime": {"model": {"path": str(args.model)}, "quant": census.quant_label,
                           "backend": "gpu", "measurement_surface": args.surface},
    }
    config = belief_context.BeliefConfig(
        root=args.belief_root_repo, ledger=args.belief_ledger,
        known_good=args.belief_known_good, escalate_after_s=args.belief_escalate_after_s,
        reader_timeout_s=args.belief_reader_timeout_s, source_roots=(args.store,),
        focus_symbols=tuple(args.belief_focus_symbol or ()))
    planner = (make_planner or _RecordingPlanner)(
        workspace=workspace, backend=backend, timeout_s=args.actor_timeout_s,
        belief_context=args.actor_belief_context, belief_root=args.belief_root_repo,
        belief=config, belief_preseeded=args.preseeded_hypotheses,
        seat=actors.ActorSeat(**seat_common))
    result, author, status, detail, code = None, None, "abstained", "", 0
    try:
        result = planner.propose(context)
        if isinstance(result, loop.Hypothesis):
            status = "authored"
            author = planner.author(result, context)
            if isinstance(author, loop.Abstain):
                status = "author_abstained"
            (out / "author.diff").write_text(_git(workspace, "diff"), encoding="utf-8")
    except belief_context.BeliefContextRefused as exc:
        status, detail, code = "belief_refused", f"{exc.reason}: {exc.detail}", 1
    except Exception as exc:  # noqa: BLE001 -- the report still records what happened
        status, detail, code = "actor_error", f"{type(exc).__name__}: {exc}"[:500], 1
    report = _report(planner, result, status=status, source_commit=source_commit,
                     profile_ref=str(profile_path), author=author, detail=detail)
    (out / REPORT_NAME).write_text(json.dumps(report, indent=2, default=str) + "\n",
                                   encoding="utf-8")
    print(f"planner-only {status}: report {out / REPORT_NAME}; stopped before build")
    return code
