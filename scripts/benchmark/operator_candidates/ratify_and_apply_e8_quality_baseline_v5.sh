#!/bin/bash
# Human-owned, fail-closed E8-v5 baseline transaction.
#
# This wrapper deliberately creates no operator receipt until the canonical
# state CAS has committed.  `--prevalidate` is read-only and prints the exact
# review material that the human must bind when later applying.
set -euo pipefail

ORCH="/mnt/raid0/llm/epyc-orchestrator"
CANONICAL_ROOT="/mnt/raid0/llm/epyc-root"
ROOT="${E8_V5_OPERATOR_ROOT:-$CANONICAL_ROOT}"
SOURCE_ROOT="${E8_V5_SOURCE_ROOT:-$ORCH}"
PYTHON="$ORCH/.venv/bin/python"
RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_v5.py"
BASE_RUNNER="$SOURCE_ROOT/scripts/benchmark/run_e8_quality_baseline_reseed.py"
PRODUCER="$SOURCE_ROOT/scripts/benchmark/terminalize_e8_quality_baseline_source.py"
RESUME_RUNNER="$SOURCE_ROOT/scripts/benchmark/resume_e8_quality_baseline_v5.py"
RECOVERY_RUNNER="$SOURCE_ROOT/scripts/benchmark/recover_e8_quality_baseline_v5_partial_r2.py"
FINALIZER_RUNNER="$SOURCE_ROOT/scripts/benchmark/finalize_e8_quality_baseline_v5_recovery_r2.py"
SUCCESSOR_RUNNER="$SOURCE_ROOT/scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_successor.py"
RACE_RETRY_RUNNER="$SOURCE_ROOT/scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_race_retry.py"
MIXED_TAIL_REPAIR_RUNNER="$SOURCE_ROOT/scripts/benchmark/prepare_e8_quality_baseline_v5_partial_r2_mixed_tail_repair.py"
TERMINALIZER_RUNNER="$SOURCE_ROOT/scripts/benchmark/terminalize_e8_quality_baseline_v5_partial_r2_successor.py"
FINAL_C1_RETRY_RUNNER="$SOURCE_ROOT/scripts/benchmark/final_c1_retry.py"
FINAL_C1_VALIDATOR="$SOURCE_ROOT/scripts/benchmark/final_c1_validator.py"
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
VALIDATOR="$SCRIPT_DIR/prepare_e8_quality_baseline_v5_candidate.sh"
VALIDATOR_PY="$SOURCE_ROOT/scripts/benchmark/validate_e8_quality_baseline_v5.py"
APPLIER="$SOURCE_ROOT/scripts/benchmark/operator_candidates/apply_e8_quality_baseline_state_v5_candidate.py"
CANONICAL_APPLIER="$CANONICAL_ROOT/artifacts/operator/apply_e8_quality_baseline_state.py"
STATE="${E8_V5_STATE:-$ORCH/orchestration/autopilot_state.json}"
LOCK="${E8_V5_LOCK_PATH:-/mnt/raid0/llm/tmp/e8-quality-baseline-v5-apply.lock}"
TRUST_LOCK="${E8_V5_TRUST_LOCK:-/run/lock/epyc-measurement-trust-boundary.lock}"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}
sha() { sha256sum -- "$1" | awk '{print $1}'; }

TEST_SANDBOX=0
override_count=0
for variable in E8_V5_OPERATOR_ROOT E8_V5_STATE E8_V5_LOCK_PATH E8_V5_TRUST_LOCK; do
  [[ -n "${!variable:-}" ]] && ((override_count += 1))
done
if ((override_count == 4)); then
  [[ "${E8_V5_TEST_MODE:-}" == "1" && -n "${PYTEST_CURRENT_TEST:-}" ]] ||
    fail 'noncanonical state/artifact paths are pytest-only'
  [[ -d "$ROOT" && -f "$STATE" && -d "$(dirname -- "$LOCK")" && -d "$(dirname -- "$TRUST_LOCK")" ]] ||
    fail 'pytest-only root, state, and lock parents must already exist'
  [[ ! -L "$ROOT" && ! -L "$STATE" && ! -L "$(dirname -- "$LOCK")" && ! -L "$LOCK" &&
  ! -L "$(dirname -- "$TRUST_LOCK")" && ! -L "$TRUST_LOCK" ]] ||
    fail 'pytest-only paths must not be symlinks'
  ROOT_LEXICAL="$(realpath -ms -- "$ROOT")"
  STATE_LEXICAL="$(realpath -ms -- "$STATE")"
  LOCK_PARENT_LEXICAL="$(realpath -ms -- "$(dirname -- "$LOCK")")"
  TRUST_LOCK_PARENT_LEXICAL="$(realpath -ms -- "$(dirname -- "$TRUST_LOCK")")"
  ROOT_RESOLVED="$(realpath -e -- "$ROOT")"
  STATE_RESOLVED="$(realpath -e -- "$STATE")"
  LOCK_PARENT_RESOLVED="$(realpath -e -- "$(dirname -- "$LOCK")")"
  TRUST_LOCK_PARENT_RESOLVED="$(realpath -e -- "$(dirname -- "$TRUST_LOCK")")"
  [[ "$ROOT_LEXICAL" == "$ROOT_RESOLVED" && "$STATE_LEXICAL" == "$STATE_RESOLVED" &&
    "$LOCK_PARENT_LEXICAL" == "$LOCK_PARENT_RESOLVED" &&
    "$TRUST_LOCK_PARENT_LEXICAL" == "$TRUST_LOCK_PARENT_RESOLVED" ]] ||
    fail 'pytest-only paths must not traverse symlinked components'
  [[ "$ROOT_RESOLVED" == /tmp/* && "$STATE_RESOLVED" == /tmp/* &&
    "$LOCK_PARENT_RESOLVED" == /tmp/* && "$TRUST_LOCK_PARENT_RESOLVED" == /tmp/* ]] ||
    fail 'pytest-only resolved paths must remain below /tmp'
  [[ "$(stat -c '%d:%i' -- "$STATE_RESOLVED")" != "$(stat -c '%d:%i' -- "$ORCH/orchestration/autopilot_state.json")" ]] ||
    fail 'pytest-only state must not resolve to the canonical production state inode'
  ROOT="$ROOT_RESOLVED"
  STATE="$STATE_RESOLVED"
  LOCK="$LOCK_PARENT_RESOLVED/$(basename -- "$LOCK")"
  TRUST_LOCK="$TRUST_LOCK_PARENT_RESOLVED/$(basename -- "$TRUST_LOCK")"
  TEST_SANDBOX=1
elif ((override_count != 0)); then
  fail 'test sandbox requires E8_V5_OPERATOR_ROOT, E8_V5_STATE, E8_V5_LOCK_PATH, and E8_V5_TRUST_LOCK'
fi

acquire_trust_boundary_lock() {
  /usr/bin/python3 - "$TRUST_LOCK" <<'PY'
import os
import stat
import sys

path = sys.argv[1]
parent = os.path.dirname(path)
nofollow = getattr(os, "O_NOFOLLOW", 0)
dirfd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | nofollow)
try:
    fd = os.open(os.path.basename(path), os.O_RDWR | os.O_CREAT | nofollow, 0o660, dir_fd=dirfd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SystemExit("measurement trust-boundary lock is not a regular file")
        os.fsync(fd)
    finally:
        os.close(fd)
    os.fsync(dirfd)
finally:
    os.close(dirfd)
PY
  exec 8<>"$TRUST_LOCK"
  /usr/bin/flock -n 8 ||
    fail "measurement trust-boundary lock is already held: $TRUST_LOCK"
  /usr/bin/python3 - "$TRUST_LOCK" "/proc/$$/fd/8" <<'PY'
import os
import stat
import sys

named = os.stat(sys.argv[1], follow_symlinks=False)
held = os.stat(sys.argv[2])
if not stat.S_ISREG(named.st_mode) or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino):
    raise SystemExit("measurement trust-boundary lock inode changed during acquisition")
PY
  if ((TEST_SANDBOX == 1)) && [[ "${E8_V5_TEST_HOLD_TRUST_LOCK_SECONDS:-0}" != 0 ]]; then
    [[ "${E8_V5_TEST_HOLD_TRUST_LOCK_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
      fail 'invalid test-only trust-lock hold duration'
    sleep "${E8_V5_TEST_HOLD_TRUST_LOCK_SECONDS}"
  fi
}

verify_reviewed_bindings() {
  for binding in \
    "E8_V5_WRAPPER_SHA256:$0" \
    "E8_V5_PRODUCER_SHA256:$PRODUCER" \
    "E8_V5_RUNNER_SHA256:$RUNNER" \
    "E8_V5_BASE_RUNNER_SHA256:$BASE_RUNNER" \
    "E8_V5_RESUME_RUNNER_SHA256:$RESUME_RUNNER" \
    "E8_V5_RECOVERY_RUNNER_SHA256:$RECOVERY_RUNNER" \
    "E8_V5_FINALIZER_RUNNER_SHA256:$FINALIZER_RUNNER" \
    "E8_V5_SUCCESSOR_RUNNER_SHA256:$SUCCESSOR_RUNNER" \
    "E8_V5_RACE_RETRY_RUNNER_SHA256:$RACE_RETRY_RUNNER" \
    "E8_V5_MIXED_TAIL_REPAIR_RUNNER_SHA256:$MIXED_TAIL_REPAIR_RUNNER" \
    "E8_V5_FINAL_C1_RETRY_RUNNER_SHA256:$FINAL_C1_RETRY_RUNNER" \
    "E8_V5_FINAL_C1_VALIDATOR_SHA256:$FINAL_C1_VALIDATOR" \
    "E8_V5_VALIDATOR_SHA256:$VALIDATOR" \
    "E8_V5_VALIDATOR_PY_SHA256:$VALIDATOR_PY" \
    "E8_V5_APPLIER_SHA256:$APPLIER" \
    "E8_V5_CANONICAL_APPLIER_SHA256:$CANONICAL_APPLIER"; do
    name="${binding%%:*}"
    path="${binding#*:}"
    expected="${!name:-}"
    [[ "$expected" =~ ^[0-9a-f]{64}$ && -f "$path" && "$(sha "$path")" == "$expected" ]] ||
      fail "reviewed artifact pin differs: $name"
  done
  if [[ -n "${E8_V5_TERMINALIZER_RUNNER_SHA256:-}" ]]; then
    [[ "$E8_V5_TERMINALIZER_RUNNER_SHA256" =~ ^[0-9a-f]{64}$ && -f "$TERMINALIZER_RUNNER" && "$(sha "$TERMINALIZER_RUNNER")" == "$E8_V5_TERMINALIZER_RUNNER_SHA256" ]] ||
      fail 'reviewed artifact pin differs: E8_V5_TERMINALIZER_RUNNER_SHA256'
  fi
  [[ "${E8_V5_ORCHESTRATOR_HEAD:-}" =~ ^[0-9a-f]{40}$ && "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" == "$E8_V5_ORCHESTRATOR_HEAD" ]] ||
    fail 'reviewed source HEAD differs from the supplied source pin'
  [[ -x "$PYTHON" && "$(readlink -f -- "$PYTHON")" == "$(readlink -f -- "$ORCH/.venv/bin/python")" ]] ||
    fail 'canonical orchestrator venv is unavailable or differs'
}

usage() {
  fail 'usage: --prevalidate|--apply|--finalize-receipt --evidence EVIDENCE --expected-pre-state-sha256 SHA --expected-candidate-state-sha256 SHA'
}

MODE="${1:-}"
[[ "$MODE" == "--prevalidate" || "$MODE" == "--apply" || "$MODE" == "--finalize-receipt" ]] || usage
[[ $# -eq 7 && "$2" == "--evidence" && "$4" == "--expected-pre-state-sha256" && "$6" == "--expected-candidate-state-sha256" ]] || usage
EVIDENCE="$3"
EXPECTED_PRE="$5"
EXPECTED_CANDIDATE="$7"
[[ "$EVIDENCE" = /* && -f "$EVIDENCE" ]] || fail 'evidence must be an existing absolute path'
[[ "$EXPECTED_PRE" =~ ^[0-9a-f]{64}$ && "$EXPECTED_CANDIDATE" =~ ^[0-9a-f]{64}$ ]] ||
  fail 'reviewed state hashes must be lowercase SHA-256'
acquire_trust_boundary_lock
verify_reviewed_bindings

# The outer campaign lock now covers validation, review, and any subsequent
# state/receipt transaction. The validator itself remains a sealed-bundle
# reader and never writes the evidence or AutoPilot state.
bash "$VALIDATOR" --validate-evidence "$EVIDENCE"

EVIDENCE_SHA256="$(sha "$EVIDENCE")"
TRANSACTION="$ROOT/artifacts/operator/e8_quality_baseline_state_v5_${EVIDENCE_SHA256}.transaction"
RECEIPT="$ROOT/artifacts/operator/e8_quality_baseline_state_v5_${EVIDENCE_SHA256}.consolidated_receipt.json"
CANONICAL_ATTESTATION="$TRANSACTION/canonical_apply_attestation.json"
REVIEW_RECORD="$ROOT/artifacts/operator/e8_quality_baseline_state_v5_${EVIDENCE_SHA256}.six_row_review.json"
[[ ! -e "$RECEIPT" ]] || fail 'a consolidated receipt already exists for this sealed evidence'
[[ "$MODE" == "--prevalidate" || "$MODE" == "--finalize-receipt" || ! -e "$TRANSACTION" ]] ||
  fail 'a transaction already exists for this sealed evidence; inspect/recover it instead'

if [[ "$MODE" != "--finalize-receipt" ]]; then
  REVIEW="$(mktemp /mnt/raid0/llm/tmp/e8-quality-v5-review.XXXXXX.json)"
  cleanup() { rm -f -- "$REVIEW"; }
  trap cleanup EXIT

  # Reconstruct and retain exactly the six state rows while the state is still
  # pre-apply.  The canonical helper independently repeats the sealed validator.
  PYTHONOPTIMIZE=0 "$PYTHON" - "$APPLIER" "$STATE" "$EVIDENCE" "$VALIDATOR" "$REVIEW" \
    "$EXPECTED_PRE" "$EXPECTED_CANDIDATE" "$MODE" <<'PY'
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

adapter_path = Path(sys.argv[1])
state_path = Path(sys.argv[2])
evidence_path = Path(sys.argv[3])
validator_path = Path(sys.argv[4])
output_path = Path(sys.argv[5])
expected_pre, expected_candidate, mode = sys.argv[6:]
spec = importlib.util.spec_from_file_location("e8_v5_consolidated_adapter", adapter_path)
if spec is None or spec.loader is None:
    raise SystemExit("ERROR: cannot import reviewed E8-v5 applier adapter")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
canonical = adapter.module

def validate() -> None:
    canonical.run_evidence_validator(validator_path, evidence_path, dict(os.environ))

if mode == "--apply" and canonical.autopilot_running():
    raise SystemExit("ERROR: AutoPilot is running; stop it before applying baseline state")
review = canonical.state_candidate_review_payload(
    state_path, evidence_path, validator_path, validate
)
if review["pre_state_sha256"] != expected_pre:
    raise SystemExit("ERROR: live pre-state differs from the reviewed pre-state")
if review["candidate_state_sha256"] != expected_candidate:
    raise SystemExit("ERROR: derived candidate differs from the reviewed candidate")
if len(review["exact_state_diff"]) != 6:
    raise SystemExit("ERROR: candidate review is not the exact six-row state diff")
output_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  if [[ "$MODE" == "--prevalidate" ]]; then
    cat "$REVIEW"
    printf 'E8 v5 prevalidation passed; no state, transaction, or receipt changed.\n'
    exit 0
  fi
fi

exec 9>"$LOCK"
flock -n 9 || fail 'another v5 apply owns the lock'
verify_reviewed_bindings

if [[ "$MODE" == "--apply" ]]; then
  [[ ! -e "$RECEIPT" && ! -e "$TRANSACTION" && ! -e "$REVIEW_RECORD" ]] ||
    fail 'state transaction, review, or receipt appeared during prevalidation; refusing to continue'
  CONFIRMATION="APPLY-E8-V5:${EVIDENCE_SHA256}:${EXPECTED_CANDIDATE}"
  PROMPT='commit the state CAS'
else
  [[ -d "$TRANSACTION" && -f "$REVIEW_RECORD" && -f "$CANONICAL_ATTESTATION" ]] ||
    fail 'receipt recovery requires a committed transaction, retained review, and canonical attestation'
  CONFIRMATION="FINALIZE-E8-V5:${EVIDENCE_SHA256}:${EXPECTED_CANDIDATE}"
  PROMPT='finalize the missing consolidated receipt without reapplying state'
fi

if ((TEST_SANDBOX == 1)) && [[ "${E8_V5_TEST_AUTO_CONFIRM:-}" == "1" ]]; then
  ANSWER="$CONFIRMATION"
else
  [[ -t 0 && -t 1 ]] || fail 'apply requires an interactive terminal confirmation'
  printf 'The sealed validator passed and the six-row candidate review is bound.\n'
  printf 'Type the following exact, transaction-specific phrase to %s:\n%s\n> ' "$PROMPT" "$CONFIRMATION"
  IFS= read -r ANSWER
fi
[[ "$ANSWER" == "$CONFIRMATION" ]] || fail 'interactive confirmation did not match; no state changed'

if [[ "$MODE" == "--apply" ]]; then
  # Persist the exact pre-apply review before the CAS so post-commit receipt
  # recovery can validate, but never regenerate, a committed candidate.
  PYTHONOPTIMIZE=0 "$PYTHON" - "$REVIEW" "$REVIEW_RECORD" <<'PY'
import os, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:3])
data = source.read_bytes()
destination.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
try:
    offset = 0
    while offset < len(data):
        written = os.write(fd, memoryview(data)[offset:])
        if written <= 0:
            raise OSError("review write made no progress")
        offset += written
    os.fsync(fd)
finally:
    os.close(fd)
dir_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
PY

  COMMON=(
    --state "$STATE"
    --evidence "$EVIDENCE"
    --canonical-evidence "$EVIDENCE"
    --validator "$VALIDATOR"
    --transaction-dir "$TRANSACTION"
    --attestation "$CANONICAL_ATTESTATION"
    --expected-pre-state-sha256 "$EXPECTED_PRE"
    --expected-candidate-state-sha256 "$EXPECTED_CANDIDATE"
  )
  # The canonical applier owns lifecycle/state locks, durable preimage,
  # evidence re-validation, CAS, rollback, and the transaction-local receipt.
  if E8_BASELINE_APPLY_TOKEN="$CONFIRMATION" EPYC_MEASUREMENT_TRUST_LOCK_FD=8 \
    PYTHONOPTIMIZE=0 "$PYTHON" "$APPLIER" \
    "${COMMON[@]}" --attest "$CONFIRMATION"; then
    :
  else
    applier_status=$?
    if [[ ! -e "$TRANSACTION" && -f "$REVIEW_RECORD" && "$(sha "$STATE")" == "$EXPECTED_PRE" ]]; then
      rm -f -- "$REVIEW_RECORD"
    fi
    exit "$applier_status"
  fi
fi

# Create the one external receipt only after the canonical commit has returned
# successfully.  A failed apply therefore cannot look ratified.
PYTHONOPTIMIZE=0 "$PYTHON" - "$APPLIER" "$STATE" "$EVIDENCE" "$VALIDATOR" "$REVIEW_RECORD" \
  "$TRANSACTION" "$CANONICAL_ATTESTATION" "$RECEIPT" "$0" "$PRODUCER" "$RUNNER" "$BASE_RUNNER" \
  "$RESUME_RUNNER" "$RECOVERY_RUNNER" "$FINALIZER_RUNNER" "$SUCCESSOR_RUNNER" "$RACE_RETRY_RUNNER" \
  "$MIXED_TAIL_REPAIR_RUNNER" "$TERMINALIZER_RUNNER" "$FINAL_C1_RETRY_RUNNER" "$FINAL_C1_VALIDATOR" "$VALIDATOR_PY" "$CANONICAL_APPLIER" \
  "$EXPECTED_PRE" "$EXPECTED_CANDIDATE" "$CONFIRMATION" <<'PY'
import hashlib
import importlib.util
import json
import sys
import atexit
from datetime import UTC, datetime
from pathlib import Path

(
    adapter_path, state_path, evidence_path, validator_path, review_path,
    transaction_path, canonical_attestation_path, receipt_path, wrapper_path,
    producer_path, runner_path, base_runner_path, resume_runner_path,
    recovery_runner_path, finalizer_runner_path, successor_runner_path,
    race_retry_runner_path, mixed_tail_repair_runner_path, terminalizer_runner_path, final_c1_retry_runner_path,
    final_c1_validator_path, validator_py_path,
    canonical_applier_path,
) = [Path(value) for value in sys.argv[1:24]]
expected_pre, expected_candidate, confirmation = sys.argv[24:27]

spec = importlib.util.spec_from_file_location("e8_v5_consolidated_receipt_adapter", adapter_path)
if spec is None or spec.loader is None:
    raise SystemExit("ERROR: cannot import reviewed E8-v5 applier adapter")
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
canonical = adapter.module

if canonical.autopilot_running():
    raise SystemExit("ERROR: AutoPilot is running; stop it before finalizing the consolidated receipt")
receipt_locks = canonical.exclusive_locks(state_path)
receipt_locks.__enter__()
atexit.register(receipt_locks.__exit__, None, None, None)

review_bytes = review_path.read_bytes()
def validate() -> None:
    canonical.run_evidence_validator(validator_path, evidence_path, dict(__import__("os").environ))

review, review_sha = canonical.validate_state_candidate_review(
    review_path,
    state_path,
    evidence_path,
    validator_path,
    validate,
    allow_applied=True,
)
if len(review["exact_state_diff"]) != 6:
    raise SystemExit("ERROR: retained review lacks the exact six-row binding")
if review["pre_state_sha256"] != expected_pre:
    raise SystemExit("ERROR: retained review pre-state differs from the supplied human binding")
if review["candidate_state_sha256"] != expected_candidate:
    raise SystemExit("ERROR: retained review candidate differs from the supplied human binding")
transaction = canonical.load_json(transaction_path / "transaction.json", "committed transaction")
if transaction.get("state") != "committed":
    raise SystemExit("ERROR: canonical transaction did not commit")
if transaction.get("state_file", {}).get("pre_sha256") != review["pre_state_sha256"]:
    raise SystemExit("ERROR: transaction pre-state differs from the six-row review")
if transaction.get("state_file", {}).get("candidate_sha256") != review["candidate_state_sha256"]:
    raise SystemExit("ERROR: transaction candidate differs from the six-row review")
canonical_attestation = canonical.load_json(
    canonical_attestation_path, "canonical transaction-local attestation"
)
if canonical_attestation.get("state_sha256") != review["candidate_state_sha256"]:
    raise SystemExit("ERROR: canonical attestation does not bind the reviewed candidate")

pin = canonical.pin_evidence(evidence_path)
pin.verify()
manifest = canonical.load_json(pin.manifest_path, "sealed evidence manifest")
protocol = manifest.get("protocol_candidate")
if not isinstance(protocol, dict) or not isinstance(protocol.get("path"), str):
    raise SystemExit("ERROR: sealed evidence lacks a protocol candidate")
protocol_path = Path(protocol["path"]).resolve(strict=True)
protocol_sha = canonical.sha256_path(protocol_path)
if protocol.get("sha256") != protocol_sha:
    raise SystemExit("ERROR: protocol candidate hash differs from sealed manifest")
protocol_document = canonical.load_json(protocol_path, "sealed protocol candidate")
protocol_id = protocol_document.get("protocol", {}).get("protocol_id")
if protocol_id != "e8_quality_full_pool_tier_baseline.v5":
    raise SystemExit("ERROR: sealed protocol candidate is not E8-v5")

sha = canonical.sha256_path
payload = {
    "schema": "epyc.operator_e8_quality_baseline_v5_consolidated_receipt.v2",
    "decision": "interactive E8-v5 state apply",
    "confirmation_binding_sha256": hashlib.sha256(confirmation.encode()).hexdigest(),
    "finalized_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    "state_review": review,
    "state_review_sha256": review_sha,
    "evidence": {
        "manifest_path": str(pin.manifest_path),
        "manifest_sha256": pin.manifest_sha256,
        "run_seal_path": str(pin.seal_path),
        "run_seal_sha256": pin.seal_sha256,
        "bundle_sha256": {str(path): digest for path, digest in sorted(pin.bundle_sha256.items())},
        "protocol_candidate_path": str(protocol_path),
        "protocol_candidate_sha256": protocol_sha,
        "protocol_id": protocol_id,
    },
    "code_sha256": {
        "wrapper": sha(wrapper_path),
        "producer": sha(producer_path),
        "runner": sha(runner_path),
        "base_runner": sha(base_runner_path),
        "resume_runner": sha(resume_runner_path),
        "recovery_runner": sha(recovery_runner_path),
        "finalizer_runner": sha(finalizer_runner_path),
        "successor_runner": sha(successor_runner_path),
        "race_retry_runner": sha(race_retry_runner_path),
        "mixed_tail_repair_runner": sha(mixed_tail_repair_runner_path),
        "final_c1_retry_runner": sha(final_c1_retry_runner_path),
        "final_c1_validator": sha(final_c1_validator_path),
        "validator_wrapper": sha(validator_path),
        "validator_python": sha(validator_py_path),
        "applier_adapter": sha(adapter_path),
        "canonical_applier": sha(canonical_applier_path),
    },
    "transaction": {
        "path": str(transaction_path.resolve()),
        "journal_sha256": sha(transaction_path / "transaction.json"),
        "canonical_attestation_path": str(canonical_attestation_path.resolve()),
        "canonical_attestation_sha256": sha(canonical_attestation_path),
    },
}
if __import__("os").environ.get("E8_V5_TERMINALIZER_RUNNER_SHA256"):
    payload["code_sha256"]["terminalizer_runner"] = sha(terminalizer_runner_path)
if receipt_path.exists():
    raise SystemExit("ERROR: a consolidated receipt already exists; refusing to replace it")
canonical.write_json_create_only(receipt_path, payload)
PY

printf 'E8 v5 state CAS committed and consolidated receipt created: %s\n' "$RECEIPT"
