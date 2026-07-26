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

The host has cgroup v2, but `/sys/fs/cgroup` is actually `node:root 0755`:
the invoking user can write and rename direct children there. This differs from
the earlier accepted assumption that the cgroup root was root-owned and
non-user-writable. Do not change global cgroup ownership or mode. Instead, the
root actor creates a unique root-owned source cgroup and bind-mounts it at the
fixed, root-owned `/run/epyc-m1-cgroup`. `/run` is root-owned and not writable
by the invoking user, so once the bind mount is in place the user cannot rename
or replace the path that the runner opens. The source name under the writable
`/sys/fs/cgroup` may be renamed after setup; that does not alter the mounted
dentry under `/run`, which is the only path accepted by the runner.

The runner never elevates. An operator with working noninteractive sudo must
perform the following exact setup before launch. Root creates the source,
target, and bind mount; it delegates only `cgroup.procs` and `cgroup.kill` on
the mounted cgroup itself. No user-writable file is executed as root.

```bash
RUN_TOKEN=$(/usr/bin/basename "$RUN_DIR")
SOURCE_CGROUP=/sys/fs/cgroup/epyc-m1-source-"$RUN_TOKEN"
CGROUP_ROOT=/run/epyc-m1-cgroup
CANDIDATE_CGROUP=$CGROUP_ROOT
CALLER_UID=$(/usr/bin/id -u)
CALLER_GID=$(/usr/bin/id -g)
SETUP_FACTS=$(
/usr/bin/sudo -n "$PYTHON" - "$SOURCE_CGROUP" "$CGROUP_ROOT" "$CALLER_UID" "$CALLER_GID" <<'PY'
import os, pathlib, stat, subprocess, sys
source, target, uid, gid = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
if source.parent != pathlib.Path('/sys/fs/cgroup') or target != pathlib.Path('/run/epyc-m1-cgroup'):
    raise SystemExit('unexpected cgroup setup path')
parent_fd = os.open('/sys/fs/cgroup', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
run_fd = os.open('/run', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
source_fd = target_fd = None
source_created = target_created = mounted = False
original = os.fstat(parent_fd)
runtime = os.fstat(run_fd)
if not stat.S_ISDIR(runtime.st_mode) or (runtime.st_uid, runtime.st_gid, stat.S_IMODE(runtime.st_mode)) != (0, 0, 0o755):
    raise SystemExit('/run must be root:root mode 0755')
if not stat.S_ISDIR(original.st_mode) or (original.st_uid, original.st_gid, stat.S_IMODE(original.st_mode)) != (uid, 0, 0o755):
    raise SystemExit('/sys/fs/cgroup ownership or mode differs from the observed host')
try:
    # The source parent is made root-only only while its child is created/opened.
    os.fchown(parent_fd, 0, 0); os.fchmod(parent_fd, 0o755)
    os.mkdir(source.name, 0o755, dir_fd=parent_fd)
    source_created = True
    source_fd = os.open(source.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    os.fchown(source_fd, 0, 0); os.fchmod(source_fd, 0o755)
    os.mkdir(target.name, 0o755, dir_fd=run_fd)
    target_created = True
    target_fd = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=run_fd)
    os.fchown(target_fd, 0, 0); os.fchmod(target_fd, 0o755)
    source_before, target_before = os.fstat(source_fd), os.fstat(target_fd)
    subprocess.run(
        ['/usr/bin/mount', '--bind', f'/proc/self/fd/{source_fd}', f'/proc/self/fd/{target_fd}'],
        check=True, pass_fds=(source_fd, target_fd),
    )
    mounted = True
    # target_fd is intentionally pre-mount; source_fd names the mounted cgroup inode.
    os.chown('cgroup.procs', uid, gid, dir_fd=source_fd, follow_symlinks=False)
    os.chown('cgroup.kill', uid, gid, dir_fd=source_fd, follow_symlinks=False)
    print(f'{source_before.st_dev}:{source_before.st_ino} {target_before.st_dev}:{target_before.st_ino}')
except BaseException as original_error:
    cleanup_errors = []
    if mounted:
        try:
            subprocess.run(['/usr/bin/umount', str(target)], check=True)
            mounted = False
        except BaseException as exc:
            cleanup_errors.append(f'umount: {exc}')
    if not mounted and target_created:
        try:
            os.rmdir(target.name, dir_fd=run_fd)
            target_created = False
        except BaseException as exc:
            cleanup_errors.append(f'target rmdir: {exc}')
    if not mounted and not target_created and source_created:
        try:
            os.rmdir(source.name, dir_fd=parent_fd)
            source_created = False
        except BaseException as exc:
            cleanup_errors.append(f'source rmdir: {exc}')
    if cleanup_errors:
        raise RuntimeError('; '.join(cleanup_errors)) from original_error
    raise
finally:
    os.fchown(parent_fd, original.st_uid, original.st_gid)
    os.fchmod(parent_fd, stat.S_IMODE(original.st_mode))
    for fd in (target_fd, source_fd, run_fd, parent_fd):
        if fd is not None: os.close(fd)
PY
)
read -r SOURCE_CGROUP_DEV_INO CGROUP_TARGET_DIR_DEV_INO <<<"$SETUP_FACTS"

CGROUP_ROOT_DEV_INO=$(/usr/bin/stat -c '%d:%i' "$CGROUP_ROOT")
test "$(/usr/bin/stat -c '%u:%g:%a' /run)" = "0:0:755"
test "$(/usr/bin/stat -c '%u:%g:%a' "$CGROUP_ROOT")" = "0:0:755"
test -w "$CGROUP_ROOT/cgroup.procs"
test -w "$CGROUP_ROOT/cgroup.kill"
```

The runner independently repeats the canonical `/run/epyc-m1-cgroup` path,
`/run` ownership/non-writability, root owner/mode/inode, exact cgroup2
mountinfo record, mount device/root/source/filesystem type, exact mount
inode/owner/mode, cgroup type,
controller, `cgroup.events` populated state, membership, `cgroup.procs`, and
`cgroup.kill` checks before forking. It holds the stable mount descriptor with
`O_NOFOLLOW` while sampling and revalidating mountinfo. Missing delegation fails before fork;
there is no PID-only fallback. No recursive ownership change is used.

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

trap - EXIT
cleanup_armed=0
target_mounted=1
target_created=1
source_created=1
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
  if [[ $target_mounted -eq 1 ]]; then
    "$PYTHON" - "$SOURCE_CGROUP" "$CGROUP_ROOT" \
      "$SOURCE_CGROUP_DEV_INO" "$CGROUP_ROOT_DEV_INO" <<'PY' || cleanup_rc=1
import os, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:3])
source_expected, target_expected = sys.argv[3:]
assert f'{source.stat().st_dev}:{source.stat().st_ino}' == source_expected
assert f'{target.stat().st_dev}:{target.stat().st_ino}' == target_expected
matches = []
for line in pathlib.Path('/proc/self/mountinfo').read_text(encoding='ascii').splitlines():
    left, right = line.split(' - ', 1)
    fields, fs = left.split(), right.split()
    if fields[4] == str(target):
        matches.append((fields[2], fields[3], fs[0], fs[1]))
assert len(matches) == 1
major_minor, mount_root, fs_type, mount_source = matches[0]
assert fs_type == 'cgroup2' and mount_source == 'cgroup'
assert major_minor == f'{os.major(target.stat().st_dev)}:{os.minor(target.stat().st_dev)}'
assert mount_root == '/' + source.name
PY
    if [[ $cleanup_rc -eq 0 ]]; then
      /usr/bin/sudo -n /usr/bin/umount -- "$CGROUP_ROOT" || cleanup_rc=$?
      [[ $cleanup_rc -eq 0 ]] && target_mounted=0
    fi
  fi
  if [[ $target_mounted -eq 0 && $target_created -eq 1 ]]; then
    test "$(/usr/bin/stat -c '%d:%i' "$CGROUP_ROOT")" = \
      "$CGROUP_TARGET_DIR_DEV_INO" || cleanup_rc=1
    test "$(/usr/bin/stat -c '%u:%g:%a' "$CGROUP_ROOT")" = "0:0:755" || cleanup_rc=1
    if [[ $cleanup_rc -eq 0 ]]; then
      /usr/bin/sudo -n /usr/bin/rmdir -- "$CGROUP_ROOT" || cleanup_rc=$?
      [[ $cleanup_rc -eq 0 ]] && target_created=0
    fi
  fi
  if [[ $target_mounted -eq 0 && $target_created -eq 0 && $source_created -eq 1 ]]; then
    test "$(/usr/bin/stat -c '%d:%i' "$SOURCE_CGROUP")" = \
      "$SOURCE_CGROUP_DEV_INO" || cleanup_rc=1
    test "$(/usr/bin/stat -c '%u:%g:%a' "$SOURCE_CGROUP")" = "0:0:755" || cleanup_rc=1
    if [[ $cleanup_rc -eq 0 ]]; then
      /usr/bin/sudo -n /usr/bin/rmdir -- "$SOURCE_CGROUP" || cleanup_rc=$?
      [[ $cleanup_rc -eq 0 ]] && source_created=0
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

cleanup_on_exit
```

Cleanup publishes a durable intent before `cgroup.kill`, waits for both empty
direct membership and `cgroup.events populated=0`, then requires the candidate
port dead and physical GPU 0 idle before publishing the receipt. The trap and
terminal cleanup verify the bound source/target identity, unmount the target, remove the inode-matched empty
target directory, and finally remove the inode-matched source. They never use
recursive removal. A source-path drift under user-writable `/sys/fs/cgroup`
fails closed and leaves it for root inspection; it cannot redirect an operation
on the stable `/run` bind mount. A concurrent root actor is outside this
runbook's unprivileged threat boundary.
