# M1 Execution Capture Runbook

M1 is an observation-only paired screen. It authorizes no lineup, registry,
deployment, promotion, purchase, or closure decision. Do not execute this
runbook until the operator accepts the capture harness.

## Prerequisites

Use `/usr/bin/python3` from the main environment. Both tools fail closed unless
Python is at least 3.13 and the required pidfd APIs exist. The orchestrator
Python 3.12 environment is not supported.

```bash
set -euo pipefail
set -C
PYTHON=/usr/bin/python3
"$PYTHON" - <<'PY'
import os
import signal
import sys
assert sys.version_info >= (3, 13)
assert hasattr(os, "pidfd_open")
assert hasattr(signal, "pidfd_send_signal")
PY
```

The runner pins frozen branch `production-consolidated-v8`, commit
`67a433bf45a8a091d83b4ea0b32ff0735fd51800`, clean tracked worktree state,
`llama-server` version `10107`, both binaries, both model/projector pairs, and
every loaded llama/ggml/mtmd shared object. No hash supplied at execution time
is trusted.

Create one canonical evidence directory. Every manifest, authority, PID file,
log, launch record, capture, scored result, paired result, cleanup intent, and
cleanup receipt must be an absolute direct child of this directory. The tools
use directory FDs and `O_NOFOLLOW`; symlinks, relative paths, and escapes fail.

```bash
M1_DIR=/mnt/raid0/llm/epyc-inference-research/artifacts/minicpm-o-phase1-v8-20260726
RUN_DIR=$(/usr/bin/mktemp -d "$M1_DIR/live-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
RUN_DIR=$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=True))' "$RUN_DIR")
cd "$M1_DIR"

"$PYTHON" m1_observation_runner.py \
  --run-dir "$RUN_DIR" \
  --write-manifests "$RUN_DIR"
```

## Privilege Boundary

The host has cgroup v2, but `/sys/fs/cgroup` is `node:root 0755` and no user
systemd bus is available. The runner never elevates. An operator with working
noninteractive sudo must create exactly one leaf cgroup before launch. These
are the only privileged setup commands:

```bash
RUN_TOKEN=$(/usr/bin/basename "$RUN_DIR")
CANDIDATE_CGROUP=/sys/fs/cgroup/epyc-m1-"$RUN_TOKEN"
CALLER_UID=$(/usr/bin/id -u)
CALLER_GID=$(/usr/bin/id -g)

/usr/bin/sudo -n /usr/bin/mkdir --mode=0700 -- "$CANDIDATE_CGROUP"
/usr/bin/sudo -n /usr/bin/chown -- "$CALLER_UID:$CALLER_GID" \
  "$CANDIDATE_CGROUP" \
  "$CANDIDATE_CGROUP/cgroup.procs" \
  "$CANDIDATE_CGROUP/cgroup.kill"
/usr/bin/sudo -n /usr/bin/chmod 0700 -- "$CANDIDATE_CGROUP"

test "$(/usr/bin/stat -c '%u:%g:%a' /sys/fs/cgroup)" = "0:0:755"
test ! -w /sys/fs/cgroup
test ! -L "$CANDIDATE_CGROUP"
test "$(/usr/bin/stat -c '%u:%g' "$CANDIDATE_CGROUP")" = "$CALLER_UID:$CALLER_GID"
test "$(/usr/bin/stat -c '%a' "$CANDIDATE_CGROUP")" = 700
test -w "$CANDIDATE_CGROUP/cgroup.procs"
test -w "$CANDIDATE_CGROUP/cgroup.kill"
CANDIDATE_CGROUP_DEV_INO=$(/usr/bin/stat -c '%d:%i' "$CANDIDATE_CGROUP")
```

The runner independently repeats canonical path, inode, owner, mode, cgroup
type, controller, `cgroup.events` populated state, membership, `cgroup.procs`,
and `cgroup.kill` checks before forking. Missing delegation fails before fork;
there is no PID-only fallback. No recursive ownership change is used. The only
root operations name the leaf directory and its two kernel control files
exactly; no user-writable script or executable runs as root.

## Candidate Lifecycle

```bash
BASE_BINARY=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
BASE_MODEL=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
BASE_MMPROJ=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf
WORKER_ENDPOINT=http://127.0.0.1:8086/v1/chat/completions
ESCALATION_ENDPOINT=http://127.0.0.1:8087/v1/chat/completions

CANDIDATE_BINARY=/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server
CANDIDATE_MODEL=/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf
CANDIDATE_MMPROJ=/mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf
CANDIDATE_PORT=19086
CANDIDATE_ENDPOINT=http://127.0.0.1:$CANDIDATE_PORT/v1/chat/completions
CANDIDATE_LOG=$RUN_DIR/candidate-server.stderr
CANDIDATE_AUTHORITY=$RUN_DIR/candidate-launch-authority.json
CLEANUP_RECEIPT=$RUN_DIR/candidate-cleanup.json
FAILURE_RECOVERY_INTENT=$CANDIDATE_AUTHORITY.failure-cleanup-intent

cleanup_armed=0
cgroup_created=1
cleanup_on_exit() {
  original_rc=$?
  trap - EXIT
  cleanup_rc=0
  cleanup_evidence=
  if [[ -s "$CANDIDATE_AUTHORITY" ]]; then
    cleanup_evidence=$CANDIDATE_AUTHORITY
  elif [[ -s "$FAILURE_RECOVERY_INTENT" ]]; then
    cleanup_evidence=$FAILURE_RECOVERY_INTENT
  fi
  if [[ $cleanup_armed -eq 1 && -n "$cleanup_evidence" &&
        ! -e "$CLEANUP_RECEIPT" ]]; then
    "$PYTHON" m1_execution_capture_runner.py \
      --run-dir "$RUN_DIR" \
      --cleanup-capture "$cleanup_evidence" \
      --cleanup-receipt "$CLEANUP_RECEIPT" \
      --timeout-seconds 30 || cleanup_rc=$?
  fi
  if [[ $cgroup_created -eq 1 ]]; then
    current_dev_ino=unavailable
    populated=unavailable
    current_dev_ino=$(/usr/bin/stat -c '%d:%i' "$CANDIDATE_CGROUP") || cleanup_rc=1
    populated=$(/usr/bin/awk '$1 == "populated" {print $2}' \
      "$CANDIDATE_CGROUP/cgroup.events") || cleanup_rc=1
    if [[ $current_dev_ino != "$CANDIDATE_CGROUP_DEV_INO" ||
          $populated != 0 || -s "$CANDIDATE_CGROUP/cgroup.procs" ]]; then
      cleanup_rc=1
    else
      /usr/bin/sudo -n /usr/bin/rmdir -- "$CANDIDATE_CGROUP" || cleanup_rc=$?
    fi
  fi
  if [[ $original_rc -eq 0 && $cleanup_rc -ne 0 ]]; then
    exit "$cleanup_rc"
  fi
  exit "$original_rc"
}
trap cleanup_on_exit EXIT

cleanup_armed=1
"$PYTHON" m1_execution_capture_runner.py \
  --run-dir "$RUN_DIR" \
  --launch-candidate "$CANDIDATE_AUTHORITY" \
  --candidate-cgroup "$CANDIDATE_CGROUP" \
  --failure-cleanup-receipt "$CLEANUP_RECEIPT" \
  --pid-file "$RUN_DIR/candidate.pid" \
  --endpoint "$CANDIDATE_ENDPOINT" \
  --mi210-load-log "$CANDIDATE_LOG" \
  --timeout-seconds 10
candidate_pid=$(<"$RUN_DIR/candidate.pid")
```

Before forking, the runner publishes a durable recovery intent that binds the
exact cgroup, endpoint, artifact paths, and failure cleanup receipt. The child
then enters the cgroup in the pre-exec hook, before candidate code can fork.
The canonical launch includes `-lv 4`; v8's default verbosity suppresses the
model and projector offload lines that the MI210 evidence grammar requires.
Authority is create-only and binds the leader PID/pidfd identity plus the
cgroup canonical path, inode, controller facts, populated state, and current
membership. Any launch-path exception after the fork boundary makes a
best-effort diagnostic publication, then uses bounded `cgroup.kill` and waits
for `populated=0` even if that diagnostic publication fails. The EXIT trap can
clean from the recovery intent when no authority was reached.

Wait for candidate health only after authority exists:

```bash
"$PYTHON" - "$CANDIDATE_PORT" <<'PY'
import http.client, sys, time
port = int(sys.argv[1])
for _ in range(300):
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        connection.request("GET", "/health")
        with connection.getresponse() as response:
            if response.status == 200:
                raise SystemExit(0)
    except Exception:
        pass
    finally:
        try:
            connection.close()
        except NameError:
            pass
    time.sleep(1)
raise SystemExit("candidate health timeout")
PY
```

Resolve existing baseline owners without changing or restarting them:

```bash
worker_pid=$("$PYTHON" -c 'import m1_execution_capture_runner as r; print(r.unique_listener_pid(8086))')
escalation_pid=$("$PYTHON" -c 'import m1_execution_capture_runner as r; print(r.unique_listener_pid(8087))')
```

## Capture

Run these four commands in order:

```bash
"$PYTHON" m1_execution_capture_runner.py --run-dir "$RUN_DIR" \
  --manifest "$RUN_DIR/m1_worker_vision_manifest.json" \
  --output "$RUN_DIR/worker-candidate-responses.json" \
  --launch-record "$RUN_DIR/worker-candidate-launch.json" \
  --launch-authority "$CANDIDATE_AUTHORITY" --mi210-load-log "$CANDIDATE_LOG" \
  --endpoint "$CANDIDATE_ENDPOINT" \
  --arm-id minicpm-o45-mi210-v8 --api-model minicpm-o-4.5 \
  --model "$CANDIDATE_MODEL" --mmproj "$CANDIDATE_MMPROJ" \
  --binary "$CANDIDATE_BINARY" --server-pid "$candidate_pid" --require-mi210

"$PYTHON" m1_execution_capture_runner.py --run-dir "$RUN_DIR" \
  --manifest "$RUN_DIR/m1_worker_vision_manifest.json" \
  --output "$RUN_DIR/worker-baseline-responses.json" \
  --launch-record "$RUN_DIR/worker-baseline-launch.json" \
  --endpoint "$WORKER_ENDPOINT" \
  --arm-id qwen25vl-worker-v8 --api-model qwen2.5-vl-7b \
  --model "$BASE_MODEL" --mmproj "$BASE_MMPROJ" \
  --binary "$BASE_BINARY" --server-pid "$worker_pid"

"$PYTHON" m1_execution_capture_runner.py --run-dir "$RUN_DIR" \
  --manifest "$RUN_DIR/m1_vision_escalation_manifest.json" \
  --output "$RUN_DIR/escalation-candidate-responses.json" \
  --launch-record "$RUN_DIR/escalation-candidate-launch.json" \
  --launch-authority "$CANDIDATE_AUTHORITY" --mi210-load-log "$CANDIDATE_LOG" \
  --endpoint "$CANDIDATE_ENDPOINT" \
  --arm-id minicpm-o45-mi210-v8 --api-model minicpm-o-4.5 \
  --model "$CANDIDATE_MODEL" --mmproj "$CANDIDATE_MMPROJ" \
  --binary "$CANDIDATE_BINARY" --server-pid "$candidate_pid" --require-mi210

"$PYTHON" m1_execution_capture_runner.py --run-dir "$RUN_DIR" \
  --manifest "$RUN_DIR/m1_vision_escalation_manifest.json" \
  --output "$RUN_DIR/escalation-baseline-responses.json" \
  --launch-record "$RUN_DIR/escalation-baseline-launch.json" \
  --endpoint "$ESCALATION_ENDPOINT" \
  --arm-id qwen25vl-escalation-v8 --api-model qwen2.5-vl-7b \
  --model "$BASE_MODEL" --mmproj "$BASE_MMPROJ" \
  --binary "$BASE_BINARY" --server-pid "$escalation_pid"
```

Each executor-only row stores the exact canonical request byte count/hash,
2xx status, exact final URL, and lossless base64 response bytes/count/hash.
Requests use a direct loopback `http.client` connection with no proxy or
redirect mechanism; every 3xx is rejected. While the response socket is still
established, the runner binds its client/server 4-tuple and server socket inode
to the exact pinned PID's descriptor table, verifies there is no other procfs
owner, and captures immediate pre-response, live-response, and post-response
process identities. Scoring reparses the raw TCP table and revalidates all
tuple, inode, exclusive-owner, URL, and identity fields. It also decodes
strict UTF-8 JSON and requires
`choices[0].message.content == raw_content`, the endpoint to equal the launch
record, and the request bytes to reproduce exactly from the canonical fixture
and run contract.

Candidate rows also store raw `rocm-smi --showpids details` and
`--showpidgpus <pid>` output, exact commands, and hashes. Scoring reparses
numeric PID/device-count/index/VRAM fields, requires only physical GPU 0,
rejects co-residency, and reparses both `GPU use (%)` and `GPU Memory Allocated
(VRAM%)` as bounded integers rather than trusting declared values. The combined
model/projector residency floor is 5,509,644,826 bytes.

## Score And Pair

```bash
for role in worker escalation; do
  if [[ $role == worker ]]; then
    manifest=$RUN_DIR/m1_worker_vision_manifest.json
  else
    manifest=$RUN_DIR/m1_vision_escalation_manifest.json
  fi
  "$PYTHON" m1_observation_runner.py --run-dir "$RUN_DIR" \
    --manifest "$manifest" --responses "$RUN_DIR/$role-baseline-responses.json" \
    --scored-out "$RUN_DIR/$role-baseline-scored.json"
  "$PYTHON" m1_observation_runner.py --run-dir "$RUN_DIR" \
    --manifest "$manifest" --responses "$RUN_DIR/$role-candidate-responses.json" \
    --scored-out "$RUN_DIR/$role-candidate-scored.json"
  "$PYTHON" m1_observation_runner.py --run-dir "$RUN_DIR" \
    --baseline-scored "$RUN_DIR/$role-baseline-scored.json" \
    --candidate-scored "$RUN_DIR/$role-candidate-scored.json" \
    --paired-out "$RUN_DIR/$role-paired.json"
done
```

Each scored artifact binds the absolute manifest and capture paths and exact
byte hashes. Pairing re-reads those contained files, revalidates the complete
launch/authority/executor/GPU chain, recomputes every score from canonical
accepted answers and raw response content, and requires the stored scored
artifact to equal the recomputation. Stored `pass` or `passed` fields are never
trusted.

## Cleanup And Teardown

Cleanup is cgroup-owned, not leader-owned. It remains effective if the leader
forks and exits while a descendant survives:

```bash
"$PYTHON" m1_execution_capture_runner.py \
  --run-dir "$RUN_DIR" \
  --cleanup-capture "$CANDIDATE_AUTHORITY" \
  --cleanup-receipt "$CLEANUP_RECEIPT" \
  --timeout-seconds 30
cleanup_armed=0

"$PYTHON" - "$CLEANUP_RECEIPT" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    receipt = json.load(handle)
assert receipt["cgroup_empty"] is True
assert receipt["candidate_cgroup"]["populated"] is False
assert receipt["candidate_cgroup"]["member_pids"] == []
assert receipt["post_cleanup_listeners"] == []
assert receipt["gpu_state_post_cleanup"]["kfd_pids"] == []
assert receipt["gpu_state_post_cleanup"]["vram_use_percent"] == 0
PY

test "$(/usr/bin/stat -c '%d:%i' "$CANDIDATE_CGROUP")" = \
  "$CANDIDATE_CGROUP_DEV_INO"
/usr/bin/sudo -n /usr/bin/rmdir -- "$CANDIDATE_CGROUP"
cgroup_created=0
trap - EXIT
```

Cleanup publishes a durable intent before `cgroup.kill`, waits for both empty
direct membership and `cgroup.events populated=0`, then requires the candidate
port dead and physical GPU 0 idle before publishing the receipt. The final
privileged action is only `rmdir` of that already-empty, inode-matched exact
cgroup. `/sys/fs/cgroup` is root-owned mode 0755 and not writable by the
unprivileged runner, so the caller cannot rename or replace the leaf between
the inode check and the exact privileged `rmdir`; a concurrent root actor is
outside this runbook's unprivileged threat boundary.
