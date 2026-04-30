#!/bin/bash
# v5 cleanup audit Phase 4 validation gates
# Run end-to-end after Batch 1 (build) completes.
set -uo pipefail   # NOTE: not -e on purpose — we want to keep going past per-gate failures so the wrap-up is complete

BUILD=/mnt/raid0/llm/llama.cpp-experimental/build_v5_clean
BIN=$BUILD/bin
BUNDLE=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-30-v5-cleanup-audit/phase4-validation-gates
mkdir -p $BUNDLE

CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
Q8=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
REAP=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf
GEMMA=/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf
WIKI=/mnt/raid0/llm/data/wiki.test.raw

export LD_LIBRARY_PATH=$BIN
ulimit -c 0

CANONICAL_ENV="OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active"
CANONICAL_PFX="numactl --interleave=all -- taskset -c 0-95"

run_canonical() {
    local LOG=$1
    shift
    echo "=== $(date) ===" >> $LOG
    echo "CMD: $CANONICAL_ENV $CANONICAL_PFX $*" >> $LOG
    eval "$CANONICAL_ENV $CANONICAL_PFX $*" >> $LOG 2>&1
    local RC=$?
    echo "rc=$RC" >> $LOG
    return $RC
}

# Batch 2: Reproducibility tripwire
echo "=== Batch 2: Reproducibility tripwire (Coder-30B Q4_K_M tg32 r=5) ==="
date
LOG=$BUNDLE/batch2-tripwire.log
run_canonical $LOG \
    $BIN/llama-bench -m $CODER -t 96 -fa 1 --mmap 0 -p 0 -n 32 -r 5
echo "  log: $LOG"

# Batch 3: PPL bit-exact gates (4 models)
echo "=== Batch 3: PPL bit-exact gates ==="
date
for entry in "coder30:$CODER" "q8:$Q8" "reap:$REAP" "gemma:$GEMMA"; do
    name=${entry%%:*}
    model=${entry#*:}
    LOG=$BUNDLE/batch3-ppl-${name}.log
    echo "  $name (chunks 1-12) → $LOG"
    date
    run_canonical $LOG \
        $BIN/llama-perplexity -m $model -f $WIKI -t 96 -fa 1 --chunks 12
done

# Batch 4: No-regression bench
echo "=== Batch 4: No-regression bench (tg32/tg64 r=5) ==="
date
for entry in "coder30:$CODER:32" "q8:$Q8:32" "reap:$REAP:32" "gemma:$GEMMA:64"; do
    name=${entry%%:*}
    model=$(echo $entry | cut -d: -f2)
    n=$(echo $entry | cut -d: -f3)
    LOG=$BUNDLE/batch4-bench-${name}.log
    echo "  $name (tg$n r=5) → $LOG"
    date
    run_canonical $LOG \
        $BIN/llama-bench -m $model -t 96 -fa 1 --mmap 0 -p 0 -n $n -r 5
done

# Batch 5: Per-role smoke — DEFERRED to manual run; orchestrator_stack.py
# would be the proper launcher and orchestrator_stack already wires the v5
# env per the deployment-draft. Recording note here rather than coding ad-hoc
# server smokes that won't reflect real production launch posture.
echo "=== Batch 5: Per-role smoke (DEFERRED — see batch5-note.md) ==="
cat > $BUNDLE/batch5-note.md <<'EOF'
# Batch 5 (per-role smoke) — Deferred

Per-role smoke needs orchestrator_stack.py to launch llama-server with the
per-role env block from `model-registry-v5-deployment-draft.yaml`. Coding
ad-hoc curl-based smokes outside the orchestrator framework would not
reflect the real production launch posture (host_prerequisites, role-specific
env, binary_path selection).

Recommendation: wire model-registry-v5-deployment-draft.yaml into
`orchestration/model_registry.yaml` AFTER Batch 4 passes, then run the
existing orchestrator_stack health-check / smoke flow on the populated
roles.

Smoke gate criteria (when run):
  - For each role in deployment-draft, launch llama-server with documented env
  - 5 prompts via curl /completion
  - Verify timings.predicted_per_second within ±5% of expected_throughput
EOF

echo "=== Phase 4 validation runs complete ==="
date
echo "Bundle: $BUNDLE"
ls -la $BUNDLE/
