#!/usr/bin/env python3
"""MF-VBS-1 — measure the verify-before-stop failure rate from existing BEP/REPL traces.

Task (handoffs/active/multi-file-coding-completion-capability.md, "Research Intake Update —
2026-09-14" section): before building any gate for "the model stops after an unexecuted edit",
measure how often that actually happens in the traces we already have. External prior: stock
OpenCode + GPT-OSS-20B-low stopped after an unexecuted edit in 110/248 voluntary stops.

Data source: data/bep_sandbox/*/results.jsonl + the per-turn traces they point to
(data/bep_sandbox/*/traces/*.jsonl). This is the ONLY corpus in either repo with file_write_safe/
FINAL per-turn granularity (verified: file_write_safe/FINAL( do not appear in the autopilot
journal or the internal_interaction_j17_ab turn logs). `INVALID-*` directories and `mode=="stub"`
rows are excluded (stub rows apply the dry-run reference solution directly — no model, no REPL,
not evidence about model behavior).

Per-trajectory (one results.jsonl row = one task x arm x block, mode=="real"):

  stop_type:
    forced_max_turns  -- answer_preview == "[Max turns (N) reached]" (turns hit the harness cap)
    forced_error      -- answer_preview contains "[ERROR" (backend/harness error, not a model choice)
    voluntary         -- everything else (episode ended before the turn cap with a non-error answer)

  For voluntary trajectories, classify by whether an edit landed and whether execution followed:
    (a) no_edit        -- FINAL with no file touched in the episode
    (b) edit_executed  -- an edit landed AND a later turn called run_shell(/run_python_code(
                           (the two execution-capable tools the REPL sandbox exposes to the model;
                           see src/repl_environment/environment.py _CHAINABLE_REPL_TOOLS)
    (c) edit_unverified -- an edit landed, no run_shell/run_python_code call anywhere in the episode,
                           and no delegates-to-user phrasing in any turn's raw_output
    (d) edit_delegated  -- same as (c) but a turn's raw_output matches the delegates-to-user text
                           heuristic (defined below); reported as a sub-count of the same failure,
                           per the task's "including the delegates-to-user variant" framing

  "Edit landed" = results.jsonl touched_files is non-empty, OR answer_preview starts with
  "Batch edit applied" (the batch-edit arm's automatic promotion message; batch-edit turns never
  call file_write_safe directly so touched_files is the only signal there), OR any trace turn has
  calls_file_write_safe==true. touched_files is filesystem ground truth and normally a superset of
  the trace-level file_write_safe flag; the OR exists because the "on" (batch-edit) arm's applied
  edits do not appear as calls_file_write_safe in the trace at all (short-circuited before code
  extraction — see src/graph/helpers.py _maybe_batch_edit_turn).

  IMPORTANT CAVEAT (auto-finalize): the harness's auto_wrap_final() (src/prompt_builders/
  code_utils.py) auto-appends FINAL(...) around a lone single-statement turn (e.g. a bare
  `file_write_safe(...)` call becomes `FINAL(file_write_safe(...))`), so the trace-level
  `has_final` field (computed on the RAW pre-wrap text) UNDER-counts actual finalizations. This
  script therefore does NOT use trace `has_final` to detect the stop; it uses the results.jsonl-
  level signal (turns < cap, non-error answer_preview), which reflects what the episode actually
  did regardless of whether FINAL was model-emitted or harness-synthesized. This means class (a)/
  (c) counts include single-turn "auto-finalized on the first write" episodes that never gave the
  model a chance to choose to verify -- flagged separately as `auto_finalized_single_turn` per
  trajectory so the report can separate "model chose not to verify" from "harness ended the
  episode before a verify turn was possible."

  delegates-to-user heuristic (class d): raw_output of ANY turn in the trajectory case-insensitively
  matches one of DELEGATES_TO_USER_PATTERNS below. Precision must be hand-checked on ~20 matches
  per the task; if the corpus has fewer than 20 matches, precision is reported as N/A with the
  actual count and every match is hand-checked instead of sampled.

Excludes forced stops from the failure-rate denominator; forced counts are reported separately.

Usage:
    python3 scripts/analysis/mf_vbs1_verify_before_stop.py [--repo-root PATH] [--out FILE]

Read-only: never writes into data/bep_sandbox.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# Execution-capable tool names the REPL sandbox exposes to the model (src/repl_environment/
# environment.py _CHAINABLE_REPL_TOOLS / _PARALLEL_MUTATION_REPL_TOOLS). subprocess/os.system are
# NOT importable (not in SAFE_IMPORT_MODULES), so these two calls are the only way a model turn
# can execute or test code inside its own episode.
EXECUTION_CALL_RE = re.compile(r"\brun_shell\s*\(|\brun_python_code\s*\(")
REPORT_SCHEMA = "mf_vbs1_verify_before_stop_report.v2"
_PRODUCER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical_sha256(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _seal_report(report: dict, repo_root: str, inputs: list[dict],
                 missing_inputs: list[str] | None = None) -> dict:
    """Bind the unchanged analysis body to the exact parsed inputs and producer bytes."""
    if not inputs:
        return report
    root = Path(repo_root).resolve()
    producer = Path(__file__).resolve()
    try:
        producer_path = producer.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("producer must be inside repo-root for a source-bound report") from exc
    manifest = sorted(inputs, key=lambda item: item["path"])
    source_sha = _PRODUCER_SOURCE_SHA256
    provenance = {
        "schema": REPORT_SCHEMA,
        "producer_path": producer_path,
        "producer_sha256": source_sha,
        "input_root": "epyc-orchestrator",
        "inputs": manifest,
        "missing_inputs": sorted(missing_inputs or []),
        "analysis_config_sha256": _canonical_sha256({}),
        "report_body_sha256": _canonical_sha256(report),
    }
    provenance["record_id"] = _canonical_sha256(provenance)
    report["native_provenance"] = provenance
    try:
        provenance["report_sha256"] = _canonical_sha256(report)
    except ValueError:
        # Empty/invalid corpora may contain the existing non-finite no-denominator sentinel.
        # Preserve that diagnostic output; it cannot become a native receipt.
        report.pop("native_provenance", None)
    return report

# Text heuristic for "delegates execution to the user": the final answer tells the user to
# run/test the code themselves instead of the model doing it. Matched case-insensitively against
# every turn's raw_output in the trajectory (not just the last turn, in case the model delegates
# mid-episode and then emits a bare FINAL("done")).
DELEGATES_TO_USER_PATTERNS = [
    r"you can (run|test|verify|execute)",
    r"please run",
    r"to verify[, ]",
    r"to test[, ]",
    r"run (the|this) (test|script|file|code)",
    r"run it (yourself|to)",
    r"should now (pass|work)",
    r"make sure to (run|test)",
    r"to confirm[, ]",
    r"execute the (test|script|file|code)",
    r"re-?run (the|this|it)",
    r"go ahead and run",
    r"feel free to (run|test)",
]
DELEGATES_TO_USER_RE = re.compile("|".join(DELEGATES_TO_USER_PATTERNS), re.IGNORECASE)


def _load_jsonl(path: str, *, input_manifest: list[dict] | None = None,
                repo_root: str | None = None,
                missing_inputs: list[str] | None = None) -> list[dict]:
    out = []
    if not os.path.exists(path):
        if input_manifest is not None and missing_inputs is not None:
            root = Path(repo_root or ".").resolve()
            source = Path(path).resolve()
            try:
                missing_inputs.append(source.relative_to(root).as_posix())
            except ValueError as exc:
                raise ValueError("missing input is outside repo-root") from exc
        return out
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    if input_manifest is not None:
        root = Path(repo_root or ".").resolve()
        source = Path(path).resolve()
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("parsed input is outside repo-root") from exc
        input_manifest.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
                               "byte_count": len(raw), "row_count": len(out),
                               "normalized_row_count": len(out)})
    return out


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None, float | None]:
    """Wilson score interval. Zero trials have unknown rates, represented as JSON nulls."""
    if n == 0:
        return (None, None, None)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (center - half) / denom
    hi = (center + half) / denom
    return (p, max(0.0, lo), min(1.0, hi))


@dataclass
class Trajectory:
    resdir: str
    task: str
    arm: str
    block: int
    turns: int
    quality_pass: bool
    touched_files: list
    answer_preview: str
    trace_path: str
    trace_turns: list = field(default_factory=list)

    stop_type: str = ""  # forced_max_turns | forced_error | voluntary
    edited: bool = False
    executed: bool = False
    delegates: bool = False
    delegate_matches: list = field(default_factory=list)
    vbs_class: str = ""  # a | b | c | d
    auto_finalized_single_turn: bool = False

    @property
    def traj_id(self) -> str:
        return f"{os.path.basename(self.resdir)}/{self.task}-{self.arm}-blk{self.block}"


def classify(t: Trajectory) -> None:
    preview = t.answer_preview or ""

    # edited/executed/delegates are corpus-wide signals (computed for EVERY trajectory,
    # including forced stops) so the report can answer "does verification ever happen in this
    # harness at all", not just "...among voluntary stops".
    any_fws = any(bool(turn.get("calls_file_write_safe")) for turn in t.trace_turns)
    batch_applied = preview.startswith("Batch edit applied")
    t.edited = bool(t.touched_files) or batch_applied or any_fws

    t.executed = any(
        EXECUTION_CALL_RE.search(turn.get("raw_output") or "") for turn in t.trace_turns
    )

    for turn in t.trace_turns:
        m = DELEGATES_TO_USER_RE.search(turn.get("raw_output") or "")
        if m:
            t.delegate_matches.append(m.group(0))
    t.delegates = bool(t.delegate_matches)

    # Harness auto-finalized a lone single-statement write turn (auto_wrap_final) without the
    # model ever seeing a turn where it could have chosen to call run_shell/run_python_code.
    if t.edited and len(t.trace_turns) == 1 and any_fws:
        t.auto_finalized_single_turn = True

    if "[Max turns" in preview:
        t.stop_type = "forced_max_turns"
        return
    if preview.strip().startswith('"[ERROR') or preview.strip().startswith("[ERROR"):
        t.stop_type = "forced_error"
        return
    t.stop_type = "voluntary"

    if not t.edited:
        t.vbs_class = "a"
    elif t.executed:
        t.vbs_class = "b"
    elif t.delegates:
        t.vbs_class = "d"
    else:
        t.vbs_class = "c"


def load_trajectories(repo_root: str, *, input_manifest: list[dict] | None = None,
                      missing_inputs: list[str] | None = None) -> list[Trajectory]:
    trajs: list[Trajectory] = []
    pattern = os.path.join(repo_root, "data", "bep_sandbox", "*", "results.jsonl")
    for results_path in sorted(glob.glob(pattern)):
        resdir = os.path.dirname(results_path)
        if "INVALID" in resdir:
            continue
        for row in _load_jsonl(str(results_path), input_manifest=input_manifest,
                               repo_root=repo_root, missing_inputs=missing_inputs):
            if row.get("mode") != "real":
                continue
            trace_info = row.get("trace") or {}
            trace_rel = trace_info.get("path")
            trace_turns = []
            if trace_rel:
                trace_turns = _load_jsonl(os.path.join(resdir, trace_rel),
                                          input_manifest=input_manifest, repo_root=repo_root,
                                          missing_inputs=missing_inputs)
            t = Trajectory(
                resdir=resdir,
                task=row.get("task", ""),
                arm=row.get("arm", ""),
                block=row.get("block", -1),
                turns=row.get("turns", -1),
                quality_pass=bool(row.get("quality_pass")),
                touched_files=row.get("touched_files") or [],
                answer_preview=row.get("answer_preview") or "",
                trace_path=trace_rel or "",
                trace_turns=trace_turns,
            )
            classify(t)
            trajs.append(t)
    return trajs


def summarize(trajs: list[Trajectory]) -> dict:
    forced_max = [t for t in trajs if t.stop_type == "forced_max_turns"]
    forced_err = [t for t in trajs if t.stop_type == "forced_error"]
    voluntary = [t for t in trajs if t.stop_type == "voluntary"]

    # Corpus-wide (forced + voluntary) execution-call and edit signals: is verification ever
    # attempted at all in this harness, independent of how the episode ended?
    any_edited_corpuswide = [t for t in trajs if t.edited]
    any_edited_and_call_bearing = [t for t in trajs if t.edited and t.executed]
    any_syntax_verified = [
        t for t in trajs if any(label in (t.answer_preview or "")
                                for label in ("sandbox-verified", "syntax-only (py_compile)"))
    ]
    any_owner_command_passed = [
        t for t in trajs if "py_compile + owner command passed" in (t.answer_preview or "")
    ]

    by_class = {c: [t for t in voluntary if t.vbs_class == c] for c in "abcd"}
    n_vol = len(voluntary)
    n_failure = len(by_class["c"]) + len(by_class["d"])  # edit landed, no execution (incl. delegated)
    n_edited = len(by_class["b"]) + len(by_class["c"]) + len(by_class["d"])

    rate_of_voluntary = wilson_ci(n_failure, n_vol)
    rate_of_edited = wilson_ci(n_failure, n_edited)

    n_edited_corpuswide = sum(1 for t in trajs if t.edited)
    n_edited_and_call_bearing = len(any_edited_and_call_bearing)
    rate_execution_after_edit_corpuswide = (
        wilson_ci(n_edited_and_call_bearing, n_edited_corpuswide)
    )

    def by_key(keyfn):
        groups: dict = {}
        for t in voluntary:
            k = keyfn(t)
            groups.setdefault(k, {"n_voluntary": 0, "n_edited": 0, "n_failure": 0, "classes": {"a": 0, "b": 0, "c": 0, "d": 0}})
            groups[k]["n_voluntary"] += 1
            groups[k]["classes"][t.vbs_class] += 1
            if t.vbs_class in ("b", "c", "d"):
                groups[k]["n_edited"] += 1
            if t.vbs_class in ("c", "d"):
                groups[k]["n_failure"] += 1
        return groups

    by_arm = by_key(lambda t: t.arm)
    by_task = by_key(lambda t: t.task)
    by_source = by_key(lambda t: os.path.basename(t.resdir))
    by_role = by_key(lambda t: "coder_escalation")  # single role in this corpus (see report)

    examples_c = [t.traj_id for t in by_class["c"]][:5]
    examples_d = [t.traj_id for t in by_class["d"]]

    auto_final_single_turn = [t.traj_id for t in voluntary if t.auto_finalized_single_turn]

    return {
        "prior_reference": {
            "source": "stock OpenCode + GPT-OSS-20B-low",
            "unexecuted_edit_before_stop": "110/248 voluntary stops",
        },
        "corpus": {
            "sources_scanned": sorted({os.path.basename(t.resdir) for t in trajs}),
            "n_total_real_trajectories": len(trajs),
            "n_forced_max_turns": len(forced_max),
            "n_forced_error": len(forced_err),
            "n_voluntary": n_vol,
            "role_model": None,
            "date_range": None,
            "scope_note": "Role, model, and timestamps are not bound by the parsed native result rows; unknown in this report.",
        },
        "classification": {
            "a_no_edit": len(by_class["a"]),
            "b_edit_then_executed": len(by_class["b"]),
            "c_edit_then_unverified": len(by_class["c"]),
            "d_edit_then_delegated_to_user": len(by_class["d"]),
            "n_edited_total_b_c_d": n_edited,
            "n_failure_c_plus_d": n_failure,
        },
        "rates": {
            "failure_over_all_voluntary_stops": {
                "numerator": n_failure, "denominator": n_vol,
                "point": rate_of_voluntary[0], "wilson_95ci_lo": rate_of_voluntary[1], "wilson_95ci_hi": rate_of_voluntary[2],
            },
            "failure_over_edited_voluntary_stops": {
                "numerator": n_failure, "denominator": n_edited,
                "point": rate_of_edited[0], "wilson_95ci_lo": rate_of_edited[1], "wilson_95ci_hi": rate_of_edited[2],
            },
            "no_execution_after_edit_corpuswide_forced_and_voluntary": {
                "description": (
                    "Among all forced and voluntary trajectories with an edit signal, the complement "
                    "of trajectories whose raw trace contains a run_shell/run_python_code call-shaped "
                    "string. This episode-level text heuristic does not prove a native tool call occurred "
                    "or that it occurred after the edit."
                ),
                "numerator": n_edited_corpuswide - n_edited_and_call_bearing, "denominator": n_edited_corpuswide,
                "point": 1 - rate_execution_after_edit_corpuswide[0] if rate_execution_after_edit_corpuswide[0] is not None else None,
                "wilson_95ci_lo_of_never_executed": 1 - rate_execution_after_edit_corpuswide[2] if rate_execution_after_edit_corpuswide[2] is not None else None,
                "wilson_95ci_hi_of_never_executed": 1 - rate_execution_after_edit_corpuswide[1] if rate_execution_after_edit_corpuswide[1] is not None else None,
                "call_bearing_edited_trajectories": n_edited_and_call_bearing,
            },
        },
        "delegates_to_user_variant": {
            "n_matches": len(by_class["d"]),
            "example_matched_text": [m for t in trajs for m in t.delegate_matches][:20],
            "precision_note": (
                "0 matches found in the parsed real trajectories -- precision is N/A, not 0/0-as-good. "
                "See report for why the harness's fixed FINAL(\"done\") / auto-wrap protocol structurally "
                "may foreclose natural-language delegation text; every parsed raw_output was scanned, "
                "but this remains a text heuristic."
            ) if len(by_class["d"]) == 0 else "hand-check the example_matched_text sample and report precision",
        },
        "execution_tool_availability": {
            "note": (
                "run_shell()/run_python_code() are real callables injected into the REPL sandbox globals "
                "(src/repl_environment/environment.py _CHAINABLE_REPL_TOOLS) -- the model COULD have executed "
                "or tested code. The counts below are corpus-wide text/answer-label heuristics over parsed "
                "real trajectories; they do not establish that a native tool call actually executed."
            ),
            "n_trajectories_total": len(trajs),
            "n_trajectories_with_any_edit": len(any_edited_corpuswide),
            "n_trajectories_with_any_execution_call_text_signal": sum(1 for t in trajs if t.executed),
            "n_trajectories_with_automatic_syntax_verify_only": len(any_syntax_verified),
            "n_trajectories_with_owner_command_passed": len(any_owner_command_passed),
            "automatic_syntax_verify_caveat": (
                "'sandbox-verified (py_compile)' is a legacy batch-edit label and 'syntax-only (py_compile)' "
                "is the current syntax-only label. Both denote an infrastructure syntax check, not a "
                "model-initiated acceptance decision. 'py_compile + owner command passed' is counted "
                "separately as an infrastructure check; neither label is treated as a model execution call."
            ),
        },
        "auto_finalize_caveat": {
            "note": (
                "auto_wrap_final() (src/prompt_builders/code_utils.py) auto-wraps a lone single-statement "
                "turn in FINAL(...), so trace-level has_final under-counts real finalizations; this script "
                "uses the results.jsonl-level stop signal instead (see module docstring). Trajectories "
                "flagged here ended after exactly one turn whose only content was a file_write_safe call -- "
                "the model was never given a second turn in which it could have chosen to verify."
            ),
            "n_auto_finalized_single_turn": len(auto_final_single_turn),
            "trajectory_ids": auto_final_single_turn,
        },
        "breakdown_by_arm": by_arm,
        "breakdown_by_task": by_task,
        "breakdown_by_source_dir": by_source,
        "breakdown_by_role": by_role,
        "examples_class_c": examples_c,
        "examples_class_d": examples_d,
        "prior_comparison": {
            "prior_rate": 110 / 248,
            "prior_numerator": 110,
            "prior_denominator": 248,
            "this_corpus_rate_over_all_voluntary": rate_of_voluntary[0],
            "this_corpus_rate_over_edited": rate_of_edited[0],
            "note": (
                "The historical comparison (110/248) is retained as context only. Current role, model, "
                "and time scope are unknown because the native result rows do not bind them."
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ap.add_argument("--out", default=None, help="write JSON result here (default: print to stdout)")
    args = ap.parse_args()

    input_manifest: list[dict] = []
    missing_inputs: list[str] = []
    trajs = load_trajectories(args.repo_root, input_manifest=input_manifest,
                              missing_inputs=missing_inputs)
    result = summarize(trajs)
    result = _seal_report(result, args.repo_root, input_manifest, missing_inputs)
    text = json.dumps(result, indent=2, sort_keys=False)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
