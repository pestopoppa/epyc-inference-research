#!/bin/bash
# kernel_eval.sh — the MI210 kernel-R&D verify harness (Phase 0 of the kernel-R&D loop).
# Analog of bench_canonical.sh, for EXPERIMENTAL GPU kernel variants on the MI210 (gfx90a/CDNA2).
#
# Given a runtime-env-toggled kernel variant, runs the campaign's proven rigor and emits ONE
# verified JSONL record: correctness-gate FIRST (lexicographic — a fast-but-wrong kernel is a
# FAIL, never ranked by speed), then alternated-A/B speed, then rocprofv2 mechanism confirmation.
#
# Every number is an OBSERVATION (no P-GPU-1 protocol exists for the GPU). This harness NEVER
# gates a keep/deploy/promote decision — it produces evidence for the loop's Pareto store, and
# the operator alone authorizes any production push (see mi210-kernel-rnd-loop-proposal.md).
#
# STATUS: VALIDATED 2026-07-04 against the async-prefetch kernel (GGML_CUDA_Q8_PREFETCH 0 vs 1,
# fork upstream-mtp-verify @7c28056b7, Qwen3.6-27B-MTP-Q8_0). Reproduced +~3% tg128 and a large
# MemUnitStalled reduction on mul_mat_vec_q8_0_prefetch. See fixes vs the first draft inline (FIX:).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# DEPRECATED 2026-08-02 — DO NOT RUN. Superseded by the AutoKernel trusted
# evaluator (handoffs/active/autokernel-research-loop.md, phase AK3).
#
# WHY THIS IS FENCED OFF RATHER THAN LEFT AVAILABLE: this harness never gated on
# coherence. It sets COH="coherent" for ANY non-empty generation, runs the
# baseline comparison only when --baseline-env is passed, and then emits
# "status":"OK" unconditionally. kernel_store.py:81 admits
# coherence in ("byte-identical","coherent") into its CORRECT-ONLY Pareto view,
# so every anchor-less run here has been entering the frontier as if it were
# verified. The store's existing rows are quarantined as legacy_unverified for
# exactly this reason; running this again would add more of them.
#
# It also gates GPU exclusivity with rocm-smi idle sensing (gpu_idle(), below),
# which the measurement protocols forbid as a substitute for a device claim, and
# it builds llama-bench argv by hand rather than through a codified recipe.
#
# Kept in the tree as the historical record of the async-prefetch validation and
# as the reference for what AK3 must replace. To run it anyway for archaeology,
# set KERNEL_EVAL_ALLOW_DEPRECATED=1 and understand that anything it emits is
# inadmissible.
# ============================================================================
if [[ "${KERNEL_EVAL_ALLOW_DEPRECATED:-0}" != "1" ]]; then
    cat >&2 <<'DEPRECATED'
kernel_eval.sh is DEPRECATED and must not be run.

It never gated on coherence: any non-empty generation is recorded "coherent",
the baseline compare is optional, and status is always "OK" — so its records
contaminate kernel_store.py's correct-only Pareto view.

Superseded by the AutoKernel trusted evaluator (phase AK3). See
handoffs/active/autokernel-research-loop.md §2 and §14.

Override for archaeology only: KERNEL_EVAL_ALLOW_DEPRECATED=1
DEPRECATED
    exit 2
fi

# ---- defaults (the campaign's proven env + gotchas) ----
BIN="${BIN:-/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin}"
ROCV2="${ROCV2:-/mnt/raid0/llm/tmp/mi210-build/campaign/prof/rocprof-prefix/opt/rocm-6.2.0/bin/rocprofv2}"
ROCV2_LIB="${ROCV2_LIB:-/mnt/raid0/llm/tmp/mi210-build/campaign/prof/rocprof-prefix/usr/lib/x86_64-linux-gnu}"
BUILD_DIR="${BUILD_DIR:-/mnt/raid0/llm/llama.cpp-experimental/build-hip}"
PROMPT="${PROMPT:-Write a Python function that merges two sorted lists into one sorted list.}"
REPS=3            # llama-bench -r
AB_ROUNDS=2       # alternated baseline/variant rounds
NGEN=128          # tg tokens (llama-bench)
COH_NGEN=160      # coherence generation tokens (llama-cli)
PROF_NGEN=32      # rocprofv2 generation tokens (short — profiling is expensive)
TARGET_KERNEL="mul_mat_vec_q"   # substring to profile for the mechanism confirm
DO_BUILD=0
OUT="/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_rnd_results.jsonl"
# per-step hard timeouts (SIGKILL) so a hung GPU step self-terminates instead of wedging the card
CORR_TIMEOUT="${CORR_TIMEOUT:-360}"
COH_TIMEOUT="${COH_TIMEOUT:-200}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-240}"
PROF_TIMEOUT="${PROF_TIMEOUT:-240}"

usage(){ echo "usage: $0 --model <gguf> --label <name> --variant-env 'VAR=1' [--baseline-env 'VAR=0'] [--target-kernel K] [--build] [--out f.jsonl]"; exit 2; }

MODEL="" ; LABEL="" ; VARIANT_ENV="" ; BASELINE_ENV=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --label) LABEL="$2"; shift 2;;
  --variant-env) VARIANT_ENV="$2"; shift 2;;
  --baseline-env) BASELINE_ENV="$2"; shift 2;;
  --target-kernel) TARGET_KERNEL="$2"; shift 2;;
  --build) DO_BUILD=1; shift;;
  --out) OUT="$2"; shift 2;;
  *) usage;;
esac; done
[ -n "$MODEL" ] && [ -n "$LABEL" ] && [ -n "$VARIANT_ENV" ] || usage
[ -f "$MODEL" ] || { echo "FATAL: model not found: $MODEL"; exit 1; }

export LD_LIBRARY_PATH="$BIN:/opt/rocm/lib:/usr/lib/x86_64-linux-gnu"
export HIP_VISIBLE_DEVICES=0
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_SHA="$(cd "$BUILD_DIR/.." && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

emit_fail(){ # $1=stage $2=detail  — lexicographic: a correctness/gpu failure is recorded, speed NOT reported
  printf '{"label":%s,"ts":"%s","git_sha":"%s","model":"%s","status":"FAIL","fail_stage":"%s","detail":%s,"observation":true}\n' \
    "$(printf '%s' "$LABEL" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" \
    "$TS" "$GIT_SHA" "$(basename "$MODEL")" "$1" \
    "$(printf '%s' "$2" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" >> "$OUT"
  echo "RESULT: FAIL ($1) — $2 — recorded to $OUT (speed NOT reported, lexicographic)"; exit 1; }

# ---- 0. GPU-idle gate (single-GPU serial discipline) ----
# rocm-smi prints "No KFD PIDs currently running" when the card is free; anything else => a PID is on it.
# FIX(draft): re-check on a busy read. rocm-smi --showpids can glitch to a non-standard read once
# under host load; refusing only after the busy reading PERSISTS avoids a transient hiccup aborting a
# ~10-min run. Still fail-closed: an idle read proceeds immediately (no added false-idle risk), and a
# genuine GPU job (seconds-to-minutes) reads busy on every retry, so it is still refused.
# NB: capture to a var and match via herestring — do NOT pipe rocm-smi into `grep -q`. Under
# `set -o pipefail`, grep -q exits at the first match and closes the pipe before rocm-smi finishes
# writing its footer, so rocm-smi dies on SIGPIPE (141) and pipefail reports the pipe as failed —
# the gate would then read BUSY on every call even on a completely idle card (draft bug).
gpu_idle(){ local out; out="$(rocm-smi --showpids 2>/dev/null || true)"; grep -qiE 'No KFD PIDs' <<<"$out"; }
if ! gpu_idle; then
  busy=1
  for _ in 1 2 3; do sleep 1; if gpu_idle; then busy=0; break; fi; done
  [ "$busy" = 0 ] || emit_fail gpu_busy "another KFD process is on the GPU (persisted across retries) — refusing to run (results would be poisoned)"
fi

# ---- 1. optional build (foreground, timed) ----
if [ "$DO_BUILD" = 1 ]; then
  echo "[build] make ggml-hip ..."; ( cd "$BUILD_DIR" && timeout 600 make ggml-hip -j8 ) > "$TMP/build.log" 2>&1 \
    || emit_fail build "$(tail -3 "$TMP/build.log")"
  # NIB2-58a: a build that ends with an unverified binary is the failure mode.
  # The 2026-07-31 incident was a HIP build silently loading the FROZEN
  # production CPU-only ggml via LD_LIBRARY_PATH. Fail the harness here, before
  # any correctness/speed number, if any ggml binary in this build dir resolves
  # another tree's libraries (rc=1) or if nothing verifiable was produced
  # (rc=2 — e.g. a fresh dir where `make ggml-hip` built only libraries).
  echo "[build] verifying ggml linkage ..."
  "$HERE/../utils/verify_build_linkage.sh" "$BUILD_DIR" > "$TMP/build_linkage.log" 2>&1 \
    || emit_fail build_linkage "$(tail -3 "$TMP/build_linkage.log")"
  cat "$TMP/build_linkage.log"
fi

# ---- 2. CORRECTNESS GATE FIRST (lexicographic) ----
# FIX(draft): wrap in a hard timeout and capture rc so a hang is reported as a FAIL, not a silent hang.
echo "[correctness] test-backend-ops -o MUL_MAT (variant env: $VARIANT_ENV)"
timeout --signal=KILL "$CORR_TIMEOUT" env $VARIANT_ENV "$BIN/test-backend-ops" -o MUL_MAT </dev/null \
  > "$TMP/tbo.log" 2>&1 && CRC=0 || CRC=$?
TBO="$(grep -oE '[0-9]+/[0-9]+ tests passed' "$TMP/tbo.log" | tail -1 || true)"
if [ -z "$TBO" ]; then
  [ "$CRC" = 137 ] && emit_fail correctness "test-backend-ops timed out after ${CORR_TIMEOUT}s (no pass line)"
  emit_fail correctness "test-backend-ops produced no pass line (rc=$CRC; see log)"
fi
PASS="${TBO%%/*}"; TOTAL="$(echo "$TBO" | grep -oE '/[0-9]+' | head -1 | tr -d /)"
[ "$PASS" = "$TOTAL" ] || emit_fail correctness "test-backend-ops MUL_MAT $TBO (kernel is numerically broken)"
echo "  MUL_MAT $TBO OK"

# ---- 3. output coherence (variant vs baseline greedy) ----
# FIX(draft): the volatile "[ Prompt: .. | Generation: .. t/s ]" perf line and "Exiting..." print to
# STDOUT on this build, so 2>/dev/null alone left them in the tail and every compare read as divergent.
# Strip them before comparing. Greedy (--top-k 1) + -st single-turn + </dev/null (non-interactive exit).
run_gen(){ timeout --signal=KILL "$COH_TIMEOUT" env $1 "$BIN/llama-cli" -m "$MODEL" -ngl 99 \
  -p "$PROMPT" -n "$COH_NGEN" --top-k 1 -st </dev/null 2>/dev/null \
  | grep -avE 't/s|Exiting' | tail -c 800; }
V_OUT="$(run_gen "$VARIANT_ENV")"
COH="coherent"
[ -n "$V_OUT" ] || COH="empty-generation"
if [ -n "$BASELINE_ENV" ] && [ -n "$V_OUT" ]; then
  B_OUT="$(run_gen "$BASELINE_ENV")"
  [ "$V_OUT" = "$B_OUT" ] && COH="byte-identical" || COH="divergent-but-check"
fi
echo "  coherence: $COH"

# ---- 4. SPEED — alternated A/B (rules out GPU-state drift) ----
# FIX(draft): parse llama-bench JSON avg_ts. The old md grep '[0-9]+\.[0-9]+ | tail -1' grabbed the
# stddev column (e.g. 0.00), not the tg mean. -p 0 skips prompt eval; -o json is stable to parse.
bench_tps(){ timeout --signal=KILL "$BENCH_TIMEOUT" env $1 "$BIN/llama-bench" -m "$MODEL" -ngl 99 \
  -n "$NGEN" -p 0 -fa 1 -r "$REPS" -o json 2>/dev/null | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    tg=[r for r in d if int(r.get("n_gen",0))>0] or d
    print(round(float(tg[0]["avg_ts"]),4))
except Exception:
    pass'; }
BREPS="" ; VREPS=""
for r in $(seq 1 "$AB_ROUNDS"); do
  if [ -n "$BASELINE_ENV" ]; then
    b="$(bench_tps "$BASELINE_ENV")"; [ -n "$b" ] || emit_fail speed "baseline llama-bench produced no tg number (round $r)"; BREPS="$BREPS $b"
  fi
  v="$(bench_tps "$VARIANT_ENV")"; [ -n "$v" ] || emit_fail speed "variant llama-bench produced no tg number (round $r)"; VREPS="$VREPS $v"
done
STATS="$(python3 - "$BASELINE_ENV" "$BREPS" "$VREPS" <<'PY'
import sys
hb=bool(sys.argv[1].strip())
b=[float(x) for x in sys.argv[2].split()]
v=[float(x) for x in sys.argv[3].split()]
vm=sum(v)/len(v)
if hb and b:
    bm=sum(b)/len(b); print(f"{bm:.2f} {vm:.2f} {100*(vm-bm)/bm:.2f}")
else:
    print(f"null {vm:.2f} null")
PY
)"
BMEAN="$(echo "$STATS" | awk '{print $1}')"; VMEAN="$(echo "$STATS" | awk '{print $2}')"; DELTA="$(echo "$STATS" | awk '{print $3}')"
echo "  speed: baseline mean=$BMEAN variant mean=$VMEAN delta=${DELTA}%  (reps b:[$BREPS] v:[$VREPS])"

# ---- 5. MECHANISM confirm (rocprofv2 — A/B, dominant target kernel) ----
# FIX(draft): (a) profile BOTH modes so a delta can be computed (draft profiled only the variant);
# (b) aggregate the MEAN over ALL matching dispatches (draft took the first row only);
# (c) the kernel is RENAMED across modes (baseline mul_mat_vec_q<(ggml_type)8..> -> variant
#     mul_mat_vec_q8_0_prefetch<..>), so compare the DOMINANT (busiest) matching kernel per mode —
#     robust to the rename — and also report the broad-substring aggregate for context;
# (d) rocprofv2 file plugin writes to a DIRECTORY via -d (draft passed the dir to -o, the filename flag);
# (e) drive generation with -st </dev/null so llama-cli exits (else it spins on interactive reverse-prompt).
MECH='{}'
if [ -x "$ROCV2" ]; then
  printf 'pmc: MemUnitStalled MemUnitBusy VALUBusy GRBM_GUI_ACTIVE\n' > "$TMP/pmc.txt"
  prof(){ LD_LIBRARY_PATH="$ROCV2_LIB:$LD_LIBRARY_PATH" timeout --signal=KILL "$PROF_TIMEOUT" \
      env $1 "$ROCV2" -i "$TMP/pmc.txt" --plugin file -d "$2" \
      "$BIN/llama-cli" -m "$MODEL" -ngl 99 -p "$PROMPT" -n "$PROF_NGEN" --top-k 1 -st </dev/null > "$2.log" 2>&1 || true; }
  prof "$VARIANT_ENV" "$TMP/pv"
  [ -n "$BASELINE_ENV" ] && prof "$BASELINE_ENV" "$TMP/pb"
  MECH="$(python3 - "$TMP" "$TARGET_KERNEL" <<'PY' 2>/dev/null || echo '{}'
import sys,glob,csv,json,os
d,k=sys.argv[1],sys.argv[2]
COLS=('MemUnitStalled','MemUnitBusy','VALUBusy')
def kname(r): return r.get('Kernel_Name') or r.get('KernelName') or ''
def load(tag):
    rows=[]
    for f in glob.glob(os.path.join(d,tag,'**','*.csv'),recursive=True):
        try: rows+=list(csv.DictReader(open(f)))
        except Exception: pass
    return rows
def means(rows):
    acc={c:[] for c in COLS}
    for r in rows:
        for c in COLS:
            v=r.get(c)
            if v not in (None,''):
                try: acc[c].append(float(v))
                except Exception: pass
    return {c:(round(sum(v)/len(v),4) if v else None) for c,v in acc.items()}
def analyze(rows):
    match=[r for r in rows if k in kname(r)]
    groups={}
    for r in match:
        groups.setdefault(kname(r).split('(')[0].strip(),[]).append(r)
    dom=max(groups,key=lambda g:len(groups[g])) if groups else None
    return {'dominant_kernel':dom,'dominant_dispatches':(len(groups[dom]) if dom else 0),
            'dominant':(means(groups[dom]) if dom else {c:None for c in COLS}),
            'aggregate_dispatches':len(match),'aggregate':means(match)}
def dpc(a,b): return {c:(round(100*(a[c]-b[c])/b[c],2) if a.get(c) and b.get(c) else None) for c in COLS}
V=analyze(load('pv'))
out={'target_substring':k,'variant':V}
if os.path.isdir(os.path.join(d,'pb')):
    B=analyze(load('pb')); out['baseline']=B
    out['delta_pct_dominant']=dpc(V['dominant'],B['dominant'])
    out['delta_pct_aggregate']=dpc(V['aggregate'],B['aggregate'])
print(json.dumps(out))
PY
)"
fi
DOM_STALL="$(printf '%s' "$MECH" | python3 -c 'import json,sys
try:
    m=json.load(sys.stdin); print(m.get("delta_pct_dominant",{}).get("MemUnitStalled"))
except Exception: print("n/a")' 2>/dev/null || echo n/a)"
echo "  mechanism ($TARGET_KERNEL): dominant-kernel MemUnitStalled delta = ${DOM_STALL}%"

# ---- 6. emit verified OBSERVATION record ----
python3 - "$OUT" "$LABEL" "$TS" "$GIT_SHA" "$MODEL" "$TBO" "$COH" "$BMEAN" "$VMEAN" "$DELTA" "$MECH" <<'PY'
import sys,json
out,label,ts,sha,model,tbo,coh,bmean,vmean,delta,mech=sys.argv[1:12]
def num(x):
    try: return float(x)
    except: return None
rec={"label":label,"ts":ts,"git_sha":sha,"model":model.split("/")[-1],"status":"OK",
     "correctness":{"test_backend_ops":tbo,"coherence":coh},
     "single_tps_baseline":num(bmean),"single_tps_variant":num(vmean),"delta_pct":num(delta),
     "mechanism":json.loads(mech) if mech.strip().startswith("{") else {},
     "observation":True,"note":"OBSERVATION only — not decision-gating; operator authorizes any prod push"}
open(out,"a").write(json.dumps(rec)+"\n")
print("RESULT: OK — recorded to",out)
PY
