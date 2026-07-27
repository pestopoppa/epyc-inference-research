#!/bin/bash
# v8 TB-6 continuation: one contiguous model-residency block.  Each static
# -np/-c cell must reload because llama-server allocates slots at startup; this
# is not client request concurrency.  Quality runs precede the grid within the
# same model block, avoiding a duplicate quality-only model load.
set -euo pipefail

ART=/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_study_v8_20260727
CANON=/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_study_20260723
RES=/mnt/raid0/llm/epyc-inference-research
BIN=/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server
PORT=18072
CORES=184-191
CGROUP_ROOT=/sys/fs/cgroup
SIDE_CGROUP="$CGROUP_ROOT/epyc-v8-gpu-sidecar"
PIN_THROUGHPUT="$RES/artifacts/architect-bench-gpu-20260720/questions_olympiadbench_hard.json"
PIN_SWE="$RES/artifacts/architect-code-eval-20260724/questions_swebench_oracle.json"
PIN_LCB="$RES/artifacts/architect-code-eval-20260724/questions_livecodebench_hard.json"
VALIDATOR="$ART/driver/aggregate_np_context_v8.py"
PREPARED="$ART/prefill_to_depth_rag.prepared.json"
TC_CACHE="$ART/driver/thinkingcap_identity_cache.json"
V8_HEAD=67a433bf45a8a091d83b4ea0b32ff0735fd51800
V8_BINARY_SHA=112c560f1c978c584a9899539851348a0ce1e05cde458061c281758aff066882

ensure_grid_contract() { # append-only migration for pre-v2 provenance
  local provenance=$base/provenance.txt expected existing_count
  case "$grid" in
    full) expected=cartesian24_v2 ;;
    a4_bridge) expected=a4_bridge11_v1 ;;
    *) echo "unsupported grid contract: $grid" >&2; return 1 ;;
  esac
  existing_count=$(grep -c '^grid_contract=' "$provenance" || true)
  if (( existing_count == 0 )); then
    if [[ "$grid" == full ]]; then
      printf '%s\n' \
        'grid_contract=cartesian24_v2' \
        'prior_shape=triangular18' \
        'new_contract=cartesian24_v2' \
        'operator_directive=2026-07-27 cartesian full-grid expansion' \
        'extension_containment=new_cells_use_hard_cgroup-v2_cpuset_184-191;per-cell-proofs-remain-authoritative' \
        >> "$provenance"
    else
      printf 'grid_contract=%s\n' "$expected" >> "$provenance"
    fi
  elif (( existing_count != 1 )) || ! grep -qx "grid_contract=$expected" "$provenance"; then
    echo "conflicting grid contract in $provenance" >&2
    return 1
  fi
}

if [[ ${1:-} == --self-test ]]; then
  set -u
  cell_local_smoke() {
    local np=$1 L=$2 ctx d
    ctx=$((np * L)); d="np${np}_L${L}"
    [[ "$ctx" == 2048 && "$d" == np1_L2048 ]]
  }
  cell_local_smoke 1 2048
  self_test_root=$(mktemp -d)
  # shellcheck disable=SC2317 # Invoked by the EXIT trap below.
  cleanup_self_test() {
    local status=$?
    unlink "$base/provenance.txt" 2>/dev/null || :
    rmdir "$base" 2>/dev/null || :
    rmdir "$self_test_root" 2>/dev/null || :
    exit "$status"
  }
  trap cleanup_self_test EXIT
  base=$self_test_root/full; grid=full
  mkdir -p "$base"
  printf 'mode=full grid=full mtp_depth=0 thinking=false\n' > "$base/provenance.txt"
  ensure_grid_contract
  ensure_grid_contract
  [[ $(grep -c '^grid_contract=' "$base/provenance.txt") == 1 ]]
  grep -qx 'prior_shape=triangular18' "$base/provenance.txt"
  grep -qx 'new_contract=cartesian24_v2' "$base/provenance.txt"
  grep -qx 'operator_directive=2026-07-27 cartesian full-grid expansion' "$base/provenance.txt"
  grep -qx 'extension_containment=new_cells_use_hard_cgroup-v2_cpuset_184-191;per-cell-proofs-remain-authoritative' "$base/provenance.txt"
  printf 'RUN_MODEL_BLOCK_SELF_TEST_OK\n'
  exit 0
fi

label=${1:?label}; model=${2:?model}; mtp_n=${3:?mtp depth}; thinking=${4:?thinking true|false}
mode=${5:-full}; grid=${6:-full}
[[ "$mode" == full || "$mode" == throughput_only ]] || { echo "invalid mode: $mode" >&2; exit 64; }
[[ "$grid" == full || "$grid" == a4_bridge ]] || { echo "invalid grid: $grid" >&2; exit 64; }
base="$ART/$label"
assert_runtime_identity() {
  local expected_sha expected_bytes expected_tensors expected_mtp
  case "$label" in
    A3_tc_thinkingcap_q8) expected_sha=efcb358ef86f07cf24bfd617a66bb0baa7220e9dd1c31b7d7beacd7b49e67d93; expected_bytes=29047082976; expected_tensors=866; expected_mtp=4 ;;
    A3_ff_fable_non_mtp_q8) expected_sha=2fff409d4a22e0cb11fb0ecfafed1c669b9808f7e6bc499036c6e85297f14f4d; expected_bytes=29787701792; expected_tensors=851; expected_mtp=0 ;;
    A3_ff_fable_mtp_q8) expected_sha=041c175f03b76adb70077ba470258f6b916ec4f5f066077377ef96396c3dd1d0; expected_bytes=30239022560; expected_tensors=866; expected_mtp=1 ;;
    Laguna_ud_iq2_gpu_dflash_off) expected_sha=1a0d44795f71044de1a9671bf70def4655f4ab7294b002263dfc8046820bfd2c; expected_bytes=37268665376; expected_tensors=814; expected_mtp=0 ;;
    A4_35b_a3b_v8_bridge) expected_sha=c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131; expected_bytes=37801097504; expected_tensors=753; expected_mtp=4 ;;
    *) echo "unknown canonical label: $label" >&2; return 1 ;;
  esac
  [[ "$mtp_n" == "$expected_mtp" ]] || { echo 'MTP identity mismatch' >&2; return 1; }
  [[ $(git -C /mnt/raid0/llm/llama.cpp symbolic-ref --short HEAD) == production-consolidated-v8 ]] || return 1
  [[ $(git -C /mnt/raid0/llm/llama.cpp rev-parse HEAD) == "$V8_HEAD" ]] || return 1
  [[ $(sha256sum "$BIN" | awk '{print $1}') == "$V8_BINARY_SHA" ]] || return 1
  python3 - "$label" "$model" "$expected_sha" "$expected_bytes" "$expected_tensors" "$PREPARED" "$TC_CACHE" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
label, raw, digest, size, tensors, prepared, cache = sys.argv[1:]
p = pathlib.Path(raw); st = p.stat()
current = {"path": str(p), "inode": st.st_ino, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
if current["bytes"] != int(size): raise SystemExit("model byte identity mismatch")
if label != "A3_tc_thinkingcap_q8":
    rows = {row.get("path"): row for row in json.loads(pathlib.Path(prepared).read_text()).get("models", [])}
    row = rows.get(str(p))
    if not row or row.get("sha256") != digest or any(row.get(k) != v for k, v in current.items()):
        raise SystemExit("prepared model identity drift")
    raise SystemExit(0)
cp = pathlib.Path(cache)
if cp.exists():
    cached = json.loads(cp.read_text())
    if cached.get("sha256") == digest and cached.get("tensors") == int(tensors) and all(cached.get(k) == v for k, v in current.items()):
        raise SystemExit(0)
with p.open("rb") as fh: actual = hashlib.file_digest(fh, "sha256").hexdigest()
if actual != digest: raise SystemExit("ThinkingCap SHA-256 mismatch")
cp.parent.mkdir(parents=True, exist_ok=True)
fd, tmp = tempfile.mkstemp(prefix=".thinkingcap-identity-", dir=cp.parent)
with os.fdopen(fd, "w") as out:
    json.dump({**current, "sha256": actual, "tensors": int(tensors)}, out, sort_keys=True); out.write("\n"); out.flush(); os.fsync(out.fileno())
os.replace(tmp, cp)
PY
}
assert_runtime_identity
mkdir -p "$base"

enter_sidecar_cpuset() {
  local mems cgroup_type actual
  [[ -w "$CGROUP_ROOT/cgroup.subtree_control" ]] || {
    echo "cgroup root is not writable; cannot enforce the GPU sidecar CPU ceiling" >&2
    return 1
  }
  if ! grep -qw cpuset "$CGROUP_ROOT/cgroup.subtree_control"; then
    printf '%s' '+cpuset' > "$CGROUP_ROOT/cgroup.subtree_control"
  fi
  mkdir -p "$SIDE_CGROUP"
  mems=$(<"$CGROUP_ROOT/cpuset.mems.effective")
  [[ -n "$mems" ]] || { echo "empty root cpuset.mems.effective" >&2; return 1; }
  printf '%s' "$mems" > "$SIDE_CGROUP/cpuset.mems"
  printf '%s' "$CORES" > "$SIDE_CGROUP/cpuset.cpus"
  cgroup_type=$(<"$SIDE_CGROUP/cgroup.type")
  if [[ "$cgroup_type" == "domain invalid" ]]; then
    printf '%s' threaded > "$SIDE_CGROUP/cgroup.type"
  elif [[ "$cgroup_type" != threaded ]]; then
    echo "unexpected GPU sidecar cgroup type: $cgroup_type" >&2
    return 1
  fi
  printf '%s' "$$" > "$SIDE_CGROUP/cgroup.threads"
  actual=$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)
  [[ "$actual" == "$CORES" ]] || {
    echo "GPU driver did not enter the hard cpuset ceiling: $actual" >&2
    return 1
  }
  {
    printf 'cgroup=%s\n' "$SIDE_CGROUP"
    printf 'type=%s\n' "$(<"$SIDE_CGROUP/cgroup.type")"
    printf 'cpus=%s\n' "$(<"$SIDE_CGROUP/cpuset.cpus.effective")"
    printf 'mems=%s\n' "$(<"$SIDE_CGROUP/cpuset.mems.effective")"
    printf 'driver_pid=%s\n' "$$"
  } > "$base/sidecar_cpuset.txt"
}
enter_sidecar_cpuset

if [[ ! -e "$base/provenance.txt" ]]; then
case "$grid" in
  full) grid_contract=cartesian24_v2 ;;
  a4_bridge) grid_contract=a4_bridge11_v1 ;;
esac
printf '%s\n' \
  'instrument=canonical np_context_study_20260723 TB-6 continuation' \
  'kernel=production-consolidated-v8 67a433bf45a8a091d83b4ea0b32ff0735fd51800' \
  'server-load-semantics=one contiguous model block; static -np/-c requires reload per cell' \
  'quality=SWE-oracle40 then LCB-hard53 before throughput grid' \
  "mode=$mode grid=$grid mtp_depth=$mtp_n thinking=$thinking" \
  "grid_contract=$grid_contract" \
  'affinity=hard cgroup-v2 cpuset 184-191 (disjoint from concurrent CPU instruments)' > "$base/provenance.txt"
fi
ensure_grid_contract

pid=''
stop_server() {
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 8); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
  pid=''
}
trap stop_server EXIT

fence_threads() { # dir
  local d=$1
  : > "$d/thread_affinity.apply.txt"
  for _ in 1 2; do
    taskset -apc "$CORES" "$pid" >> "$d/thread_affinity.apply.txt" 2>&1
    sleep 1
  done
  python3 - "$pid" "$CORES" > "$d/thread_affinity.json" <<'PY'
import json
import sys
from pathlib import Path

pid, expected = sys.argv[1], sys.argv[2]
task_dir = Path("/proc") / pid / "task"
before = sorted(path.name for path in task_dir.iterdir() if path.name.isdigit())
rows = []
for tid in before:
    status = (task_dir / tid / "status").read_text()
    affinity = next(
        line.split(":", 1)[1].strip()
        for line in status.splitlines()
        if line.startswith("Cpus_allowed_list:")
    )
    rows.append({"tid": int(tid), "cpus_allowed_list": affinity})
after = sorted(path.name for path in task_dir.iterdir() if path.name.isdigit())
payload = {
    "pid": int(pid),
    "expected": expected,
    "thread_count": len(rows),
    "stable_thread_set": before == after,
    "rows": rows,
}
print(json.dumps(payload, indent=2, sort_keys=True))
if before != after or not rows or any(row["cpus_allowed_list"] != expected for row in rows):
    raise SystemExit(1)
PY
}

launch() { # dir np ctx extra...
  local d=$1 np=$2 ctx=$3; shift 3
  mkdir -p "$d"
  stop_server
  printf '%q ' env GGML_IQK=1 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin taskset -c "$CORES" \
    "$BIN" -m "$model" --host 127.0.0.1 --port "$PORT" --metrics --slots --jinja --device ROCm0 -ngl all -fa on \
    -np "$np" -c "$ctx" -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 "$@" > "$d/server.argv"
  printf '\n' >> "$d/server.argv"
  env GGML_IQK=1 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin taskset -c "$CORES" \
    "$BIN" -m "$model" --host 127.0.0.1 --port "$PORT" --metrics --slots --jinja --device ROCm0 -ngl all -fa on \
    -np "$np" -c "$ctx" -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 "$@" >"$d/server.stdout" 2>"$d/server.stderr" &
  pid=$!; echo "$pid" > "$d/server.pid"
  [[ $(<"/proc/$pid/cgroup") == "0::/${SIDE_CGROUP##*/}" ]] || {
    echo "GPU server did not inherit the hard cpuset cgroup" >&2
    return 2
  }
  local deadline=$(( $(date +%s) + 600 ))
  while (( $(date +%s) < deadline )); do
    kill -0 "$pid" 2>/dev/null || { tail -n 80 "$d/server.stderr"; return 1; }
    if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -qi ok; then
      fence_threads "$d" || return 2
      return 0
    fi
    sleep 3
  done
  return 1
}

run_quality() { # suite n max questions
  local suite=$1 n=$2 max=$3 questions=$4 d
  d="$base/quality_$suite"
  mkdir -p "$d"
  local reason=(--no-enable-thinking)
  [[ "$thinking" == true ]] && reason=(--enable-thinking)
  if python3 "$VALIDATOR" --validate-quality "$d" \
      --quality-label "$label" --quality-model "$model" --quality-suite "$suite" \
      --quality-n "$n" --quality-max-tokens "$max" --quality-questions "$questions" \
      --quality-thinking "$thinking" >/dev/null 2>&1; then
    return 0
  fi
  local spec=()
  (( mtp_n > 0 )) && spec=(--spec-type draft-mtp --spec-draft-n-max "$mtp_n")
  launch "$d" 1 49152 --reasoning on --reasoning-budget 1024 \
    --reasoning-budget-message "" --reasoning-format deepseek "${spec[@]}"
  (cd "$RES" && HF_HOME=/mnt/raid0/llm/cache/huggingface RUNNER_REQUEST_TIMEOUT_S=5400 \
    uv run --offline --locked python scripts/benchmark/v7_quality_gate_runner.py \
      --host 127.0.0.1 --port "$PORT" --output "$d/summary.json" --suites "$suite" --n "$n" --limit "$n" --seed 42 \
      --max-tokens "$max" --endpoint chat --kernel production-consolidated-v8 --concurrency 1 --repeats 1 \
      --arm "${label}_rb1024_${suite}" --binary "$BIN" --models "$model" --temperature 0.6 --top-p 0.95 --top-k 20 \
      "${reason[@]}" --questions-in "$questions" --per-question-out "$d/per_question.jsonl") \
      >"$d/runner.stdout" 2>"$d/runner.stderr"
  stop_server
  python3 "$VALIDATOR" --validate-quality "$d" \
    --quality-label "$label" --quality-model "$model" --quality-suite "$suite" \
    --quality-n "$n" --quality-max-tokens "$max" --quality-questions "$questions" \
    --quality-thinking "$thinking" >/dev/null || {
      echo "quality capture failed exact validation: $suite" >&2; return 1;
    }
}

cell() { # np length
  local np=$1 L=$2 ctx d
  ctx=$((np * L))
  d="$base/np${np}_L${L}"
  mkdir -p "$d"
  if [[ -e "$d/results.json" || -e "$d/skip.txt" ]]; then
    python3 "$VALIDATOR" --root "$ART" --validate-cell "$label" "$np" "$L" >/dev/null || {
      echo "invalid existing cell disposition: $d" >&2; return 1;
    }
    return 0
  fi
  local spec=()
  (( mtp_n > 0 )) && spec=(--spec-type draft-mtp --spec-draft-n-max "$mtp_n")
  local launch_status
  if launch "$d" "$np" "$ctx" --reasoning off "${spec[@]}"; then
    :
  else
    launch_status=$?
    stop_server
    if (( launch_status != 1 )); then
      echo "server launch failed outside startup capacity detection: $d" >&2
      return 1
    fi
    local capacity_signature
    capacity_signature=$(capacity_start_signature "$d/server.stderr" || true)
    if [[ -z "$capacity_signature" ]]; then
      echo "server failed to start without a recognized capacity signature: $d" >&2
      return 1
    fi
    printf 'SKIP capacity_start signature=%s stderr_sha256=%s server_argv_sha256=%s requested_np=%s requested_L=%s requested_ctx=%s\n' \
      "$capacity_signature" "$(sha256sum "$d/server.stderr" | awk '{print $1}')" \
      "$(sha256sum "$d/server.argv" | awk '{print $1}')" "$np" "$L" "$ctx" > "$d/skip.txt"
    python3 "$VALIDATOR" --root "$ART" --validate-cell "$label" "$np" "$L" >/dev/null || {
      echo "capacity-start skip failed exact validation: $d" >&2; return 1;
    }
    return 0
  fi
  local nslot vram
  nslot=$(grep -oiP 'n_ctx_per_seq\s*=\s*\K[0-9]+|n_ctx_slot\s*=\s*\K[0-9]+' "$d/server.stderr" | tail -1 || true)
  vram=$(( $(rocm-smi --showmeminfo vram 2>/dev/null | grep -oiP 'used.*?:\s*\K[0-9]+' | head -1 || echo 0) / 1073741824 ))
  if [[ ${nslot:-0} -lt $L || $vram -gt 61 ]]; then
    printf 'SKIP n_ctx_slot=%s vram=%sG requested_L=%s\n' "${nslot:-0}" "$vram" "$L" > "$d/skip.txt"
    stop_server
    python3 "$VALIDATOR" --root "$ART" --validate-cell "$label" "$np" "$L" >/dev/null
    return 0
  fi
  (cd "$RES" && HF_HOME=/mnt/raid0/llm/cache/huggingface RUNNER_REQUEST_TIMEOUT_S=5400 \
    uv run --offline --locked python scripts/benchmark/v7_quality_gate_runner.py \
      --host 127.0.0.1 --port "$PORT" --output "$d/results.json" --suites olympiadbench_hard --n 155 --limit "$np" --seed 42 \
      --max-tokens "$L" --repeats 1 --concurrency "$np" --temperature 0.6 --top-p 0.95 --top-k 20 --no-enable-thinking \
      --endpoint chat --kernel production-consolidated-v8 --arm "${label}_np${np}_L${L}" --binary "$BIN" --models "$model" \
      --questions-in "$PIN_THROUGHPUT" --per-question-out "$d/per_question.jsonl") >"$d/runner.stdout" 2>"$d/runner.stderr"
  stop_server
  python3 "$VALIDATOR" --root "$ART" --validate-cell "$label" "$np" "$L" >/dev/null
}

capacity_start_signature() { # server.stderr
  local stderr=$1
  [[ -f "$stderr" ]] || return 1
  if grep -qiE '\bhipErrorOutOfMemory\b' "$stderr"; then
    printf '%s\n' hip_error_out_of_memory
  elif grep -qiE 'failed to allocate .*\b(HIP|ROCm|VRAM|GPU|device|KV|buffer)\b' "$stderr"; then
    printf '%s\n' allocation_failure
  elif grep -qiE '\b(HIP|ROCm)\b.*\b(out of memory|memory allocation)\b' "$stderr"; then
    printf '%s\n' rocm_memory_allocation_failure
  else
    return 1
  fi
}

if [[ "$mode" == full ]]; then
  run_quality swebench_oracle 40 3072 "$PIN_SWE"
  run_quality livecodebench_hard 53 4096 "$PIN_LCB"
fi
for np in 1 2 4 8 16 32; do cell "$np" 2048; done
if [[ "$grid" == full ]]; then
  for L in 8192 16384 32768; do
    for np in 1 2 4 8 16 32; do cell "$np" "$L"; done
  done
else
  for np in 1 2 4 8 16; do cell "$np" 8192; done
fi
python3 "$VALIDATOR" --root "$ART" --label "$label" --require-cells >/dev/null || {
  echo "refusing COMPLETE: invalid canonical surface" >&2; exit 1;
}
printf 'COMPLETE %s\n' "$(date -u +%FT%TZ)" > "$base/complete.txt"
python3 "$VALIDATOR" --root "$ART" --label "$label" --require-terminal >/dev/null
