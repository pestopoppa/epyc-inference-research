#!/bin/bash
# Shared launch/wait/kill helpers for the architect-bench GPU arms.
#
# PROMOTED INTO THE REPO 2026-07-29 by `auditor` from a session scratchpad, per
# `epyc-root/handoffs/active/architect-model-selection-bench.md` § "Follow-up tooling"
# ("so the runbook's launch pattern is executable, not just documented", runbook §10).
#
# PROVENANCE, because two copies existed and they differ on a value that matters:
#   * scratchpad copy (another session's /tmp, now ephemeral) had `CORES=88-95`;
#   * `artifacts/np_context_study_20260723/driver/gpu_lib.sh` (2026-07-24) had 184-191.
# THIS FILE TAKES 184-191. 88-95 is the SUPERSEDED pinning — the MI210 at
# 0000:43:00.0 is numa_node=1 and the host threads belong on its SMT siblings; the
# 88-95 value predates that correction. Promoting the scratchpad copy verbatim would
# have enshrined a known-wrong pinning as the repo's canonical driver.
#
# `set -uo pipefail` WITHOUT `-e`, deliberately, against the usual house style: the
# callers capture the runner's exit code and then call `gpu_kill`. Under `-e` a failed
# runner aborts before cleanup and leaves an orphaned llama-server holding the GPU,
# which is worse than the failure it would be reacting to.
set -uo pipefail

BIN="${GPU_BENCH_BIN:-/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server}"
PORT="${GPU_BENCH_PORT:-18072}"
CORES="${GPU_BENCH_CORES:-184-191}"   # node-3 SMT siblings; keeps the GPU server off
                                      # the CPU inference stack. See the note above.

gpu_launch() {  # gpu_launch <logdir> <model> <extra flags...>
  local logdir="$1"; shift
  local model="$1"; shift
  mkdir -p "$logdir"
  nohup taskset -c "$CORES" env \
    LD_LIBRARY_PATH="$(dirname "$BIN")" GGML_IQK=1 \
    ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
    "$BIN" -m "$model" --host 127.0.0.1 --port "$PORT" \
    --metrics --slots --jinja --reasoning off --device ROCm0 -ngl all -fa on \
    "$@" > "$logdir/server.stdout" 2> "$logdir/server.stderr" &
  echo $! > "$logdir/server.pid"
  printf '%s ' "$BIN" -m "$model" "$@" > "$logdir/server_command.txt"
}

gpu_wait() {  # gpu_wait <logdir> <timeout_s>
  local logdir="$1"; local timeout="${2:-600}"; local pid; pid=$(cat "$logdir/server.pid")
  local deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! ps -p "$pid" >/dev/null 2>&1; then echo "SERVER_DIED"; return 1; fi
    if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -qi ok; then echo "HEALTHY"; return 0; fi
    sleep 3
  done
  echo "TIMEOUT"; return 1
}

gpu_kill() {  # gpu_kill <logdir>
  local logdir="$1"; local pid; pid=$(cat "$logdir/server.pid" 2>/dev/null || echo "")
  [ -z "$pid" ] && return 0
  kill -TERM "$pid" 2>/dev/null; sleep 8
  if ps -p "$pid" >/dev/null 2>&1; then kill -9 "$pid" 2>/dev/null; sleep 5; fi
  if ps -p "$pid" >/dev/null 2>&1; then echo "KILL_FAILED $pid"; return 1; fi
  echo "dead $pid"; return 0
}

# THE KERNEL LABEL IS STAMPED INTO result.json AND IS NEVER GUESSED HERE.
#
# Both promoted runners hardcoded `--kernel production-consolidated-v7`. Production
# was frozen at **production-consolidated-v8** on 2026-07-25 (epyc-root CLAUDE.md),
# and `v7_quality_gate_runner.py --kernel` is free-text metadata with a
# `v7-candidate` default that nothing validates — so a promoted-verbatim script would
# have stamped v7 provenance onto results produced by the v8 binary. That is a false
# attestation in exactly the sense MEASUREMENT.md cares about, and the runner cannot
# catch it.
#
# So the label is REQUIRED and unset is a refusal, not a default. Fail closed.
gpu_require_kernel_label() {
  if [ -z "${KERNEL_LABEL:-}" ]; then
    echo "REFUSING: KERNEL_LABEL is unset. It is stamped into result.json as the" >&2
    echo "  kernel provenance of every row this run produces, and guessing it writes a" >&2
    echo "  false attestation that no downstream consumer can detect." >&2
    echo "  Set it to the kernel the binary at \$GPU_BENCH_BIN actually is, e.g.:" >&2
    echo "    KERNEL_LABEL=production-consolidated-v8 $0 ..." >&2
    echo "  Verify with: \$GPU_BENCH_BIN --version" >&2
    return 3
  fi
  return 0
}
