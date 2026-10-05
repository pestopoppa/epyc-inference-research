#!/bin/bash
# Human-owned apply-time ratification for the EvalTower resource-lane instrument.
set -euo pipefail

ORCH="/mnt/raid0/llm/epyc-orchestrator"
ROOT="/mnt/raid0/llm/epyc-root"
STATE="$ORCH/orchestration/autopilot_state.json"
ERAS="$ORCH/orchestration/instrument_eras.yaml"
RECEIPT="$ROOT/artifacts/operator/ratify_eval_resource_lanes_v2_20260805.json"
STATE_BACKUP="$ROOT/artifacts/operator/autopilot_state.pre-resource-lanes-v2-20260805.json"
TRUST_LOCK="/run/lock/epyc-measurement-trust-boundary.lock"
AUTOPILOT_LOCK="$ORCH/orchestration/.autopilot.lock"
PYTHON="$ORCH/.venv/bin/python"

BOUNDARY_ISO="2026-08-05T07:55:59Z"
BOUNDARY_EPOCH="1785916559"
POLICY="task_rate_4d_v2_resource_lanes"
EXECUTION_ID="resource_lanes_v2_prompt_load"
SCORING_ID="model_judge_tail_v1"
QUALITY_ERA="E9-eval-resource-lanes-quality"
SPEED_ERA="E9-autopilot-resource-lanes-speed"
MODE="${1:-apply}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ "$(id -u)" -ne 0 ]] || fail "run as the normal operator account, not root"
[[ "$MODE" == "apply" || "$MODE" == "--prevalidate" ]] ||
  fail "usage: $0 [--prevalidate]"
[[ -x "$PYTHON" && -f "$STATE" && -f "$ERAS" ]] || fail "canonical files are unavailable"
mkdir -p -- "$(dirname -- "$RECEIPT")"

exec 8>"$TRUST_LOCK"
flock -n 8 || fail "measurement trust boundary is busy"
exec 9>"$AUTOPILOT_LOCK"
flock -n 9 || fail "AutoPilot is running; stop it before applying the boundary"

"$PYTHON" - "$STATE" "$ERAS" "$RECEIPT" "$STATE_BACKUP" "$0" \
  "$BOUNDARY_ISO" "$BOUNDARY_EPOCH" "$POLICY" "$EXECUTION_ID" "$SCORING_ID" \
  "$QUALITY_ERA" "$SPEED_ERA" "$MODE" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

(
    state_path_s,
    eras_path_s,
    receipt_path_s,
    backup_path_s,
    script_path_s,
    boundary_iso,
    boundary_epoch_s,
    policy,
    execution_id,
    scoring_id,
    quality_era,
    speed_era,
    mode,
) = sys.argv[1:]
state_path = Path(state_path_s)
eras_path = Path(eras_path_s)
receipt_path = Path(receipt_path_s)
backup_path = Path(backup_path_s)
script_path = Path(script_path_s).resolve()
boundary_epoch = float(boundary_epoch_s)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


state_before_bytes = state_path.read_bytes()
eras_before_bytes = eras_path.read_bytes()
state = json.loads(state_before_bytes)
registry = yaml.safe_load(eras_before_bytes) or {}
era_rows = registry.get("eras") or []
existing_ids = {str(row.get("id")) for row in era_rows if isinstance(row, dict)}
present = {quality_era, speed_era} & existing_ids
if present and present != {quality_era, speed_era}:
    raise SystemExit(f"ERROR: partial E9 registry apply detected: {sorted(present)}")

if receipt_path.exists():
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") == "ratified_and_applied"
        and state.get("pareto_objective_policy") == policy
        and state.get("eval_execution_instrument_id") == execution_id
        and present == {quality_era, speed_era}
    ):
        print(f"already applied: {receipt_path}")
        raise SystemExit(0)
    raise SystemExit(f"ERROR: receipt already exists but canonical state is inconsistent: {receipt_path}")

old_policy = str(state.get("pareto_objective_policy") or "")
if old_policy not in {"task_rate_4d_v1", policy}:
    raise SystemExit(f"ERROR: unexpected pre-apply objective policy: {old_policy!r}")
if state.get("eval_execution_instrument_id") not in {None, "", execution_id}:
    raise SystemExit("ERROR: unexpected pre-apply eval_execution_instrument_id")

if not present:
    era_block = f'''

  - id: {quality_era}
    from: "{boundary_iso}"
    scope: eval_quality
    policy_version: "{policy}"
    execution_instrument_id: "{execution_id}"
    scoring_schedule_id: "{scoring_id}"
    note: >
      EvalTower resource-lane execution boundary. Prompt-weighted per-question admission
      replaces role-cached load, and model-backed llm_judge/rubric scoring runs as a
      certified scorer tail after generation lanes drain. LLM-judge transport, HTTP,
      backend, JSON, and response-shape failures retain exact error categories. Pre-boundary
      reliability/error-exclusion and aggregate quality are historical priors; preserve the
      journal and rebaseline within this era from banked outputs where replay is possible.

  - id: {speed_era}
    from: "{boundary_iso}"
    scope: autopilot_speed
    policy_version: "{policy}"
    execution_instrument_id: "{execution_id}"
    scoring_schedule_id: "{scoring_id}"
    note: >
      Questions/hour denominator boundary for certified per-resource eval lanes. v1 serial
      and v2 resource-lane task rates are not comparable even though both objective vectors
      are 4D. Retire the pre-boundary frontier view and rebuild from post-boundary trials;
      do not rescale or merge v1 snapshots into the v2 frontier.
'''
    marker = b"\nknown_dead_instrument_items:"
    before_dead, separator, after_dead = eras_before_bytes.partition(marker)
    if not separator:
        raise SystemExit("ERROR: instrument-era registry lacks known-dead section marker")
    eras_after_bytes = (
        before_dead.rstrip()
        + era_block.encode()
        + b"\n\nknown_dead_instrument_items:"
        + after_dead
    )
    # Parse the exact candidate before crossing the trust boundary.
    parsed = yaml.safe_load(eras_after_bytes) or {}
    parsed_ids = {
        str(row.get("id")) for row in (parsed.get("eras") or []) if isinstance(row, dict)
    }
    if not {quality_era, speed_era} <= parsed_ids:
        raise SystemExit("ERROR: candidate era registry failed validation")
else:
    eras_after_bytes = eras_before_bytes

active = dict(state.get("active_instrument_eras") or {})
active["eval_quality"] = quality_era
active["autopilot_speed"] = speed_era
state["active_instrument_eras"] = active
state["eval_execution_instrument_id"] = execution_id
state["eval_scoring_schedule_id"] = scoring_id
state["pareto_objective_policy"] = policy
state["pareto_objective_policy_note"] = (
    "E9 resource-lane boundary: task_rate uses certified per-resource generation lanes, "
    "prompt-weighted admission, and a model-judge scorer tail. v1 and v2 rates do not mix."
)
state["pareto_epoch_ts"] = boundary_epoch
state["pareto_exclude_before_ts"] = boundary_epoch
state["pareto_pre_epoch_speed_factor"] = 1.0
state["pareto_epoch_opened_at"] = boundary_iso
state["pareto_epoch_reason"] = f"{speed_era}: resource-lane questions/hour boundary"
state["quality_epoch_ts"] = boundary_epoch
state["quality_exclude_before_ts"] = boundary_epoch
state.pop("pareto_archive", None)
state["_allow_empty_frontier_rebase"] = True
state["_allow_empty_frontier_rebase_note"] = (
    "Operator-ratified E9 v2 objective boundary; the empty post-boundary frontier is "
    "intentional until the first resource-lane trial lands."
)
state["eval_instrument_empty_frontier_bootstrap"] = {
    "status": "pending",
    "opened_at": boundary_iso,
    "objective_policy": policy,
    "execution_instrument_id": execution_id,
    "completion_condition": "first post-boundary v2 Pareto point is reconstructed",
}
state["frontier_rerun_required"] = {
    "required": True,
    "opened_at": boundary_iso,
    "rerun_started_at": boundary_iso,
    "completed_numeric_trials": 0,
    "min_numeric_trials": 16,
    "reason": (
        f"{speed_era} opened; rebuild a v2 resource-lane frontier before using speed maxima"
    ),
    "minimum_action": (
        "Run at least 16 completed current-era numeric_trial rows, then rebuild and inspect "
        "the v2-only frontier before clearing this marker."
    ),
    "previous_marker": state.get("frontier_rerun_required"),
}

state_after_bytes = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()

if mode == "--prevalidate":
    print(json.dumps({
        "status": "prevalidated",
        "writes_performed": False,
        "boundary": boundary_iso,
        "objective_policy": policy,
        "execution_instrument_id": execution_id,
        "scoring_schedule_id": scoring_id,
        "eras_to_append": [] if present else [quality_era, speed_era],
        "state_preimage_sha256": sha(state_before_bytes),
        "state_candidate_sha256": sha(state_after_bytes),
        "eras_preimage_sha256": sha(eras_before_bytes),
        "eras_candidate_sha256": sha(eras_after_bytes),
    }, indent=2, sort_keys=True))
    raise SystemExit(0)

if not backup_path.exists():
    atomic_write(backup_path, state_before_bytes, 0o444)
if not present:
    atomic_write(eras_path, eras_after_bytes)
atomic_write(state_path, state_after_bytes)

receipt = {
    "schema_version": "eval-resource-lanes-ratification.v1",
    "status": "ratified_and_applied",
    "ratified_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "operator_uid": os.getuid(),
    "boundary": {"iso": boundary_iso, "epoch": boundary_epoch},
    "objective_policy": policy,
    "execution_instrument_id": execution_id,
    "scoring_schedule_id": scoring_id,
    "eras": {"eval_quality": quality_era, "autopilot_speed": speed_era},
    "sha256": {
        "state_preimage": sha(state_before_bytes),
        "state_applied": sha(state_after_bytes),
        "eras_preimage": sha(eras_before_bytes),
        "eras_applied": sha(eras_after_bytes),
        "ratifier": sha(script_path.read_bytes()),
    },
    "state_backup": str(backup_path),
    "autopilot_started": False,
}
atomic_write(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(), 0o444)
print(f"ratified and applied: {receipt_path}")
print("AutoPilot remains stopped; run the post-apply readiness check before starting it.")
PY
