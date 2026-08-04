#!/bin/bash
# Fixed-work KL: the review's best idea, tested on hardware.
#
# THE POINT. Throughput can be reward-hacked — deleting the computation is the
# fastest kernel there is, which is why AutoKernel needed a whole correctness
# plane. autoresearch needs none, because `val_bpb` gets WORSE if you delete work:
# correctness is intrinsic to its metric.
#
# KL-to-anchor is our val_bpb. Dump the anchor's logits once; score every candidate
# against them on the SAME fixed token set. A kernel that skips work gets a
# catastrophic KL. A kernel that is faster at equal KL is a real win. One number
# carries both, so the accept rule stops needing a separate correctness gate to
# rank on.
#
# The eval model is deliberately MoE (Qwen3-Coder-30B-A3B): MUL_MAT_ID — the expert
# dispatch our production worker runs on EVERY token — is then exercised by
# construction rather than by a gate someone has to remember to wire. The
# predecessor harness `kernel_eval.sh` tested MUL_MAT only, so a kernel that broke
# MoE dispatch passed it cleanly.
#
# What this does NOT replace, and why the review is right to say so:
#   - pairing: thermal drift hits KL-per-second exactly as hard as it hits t/s
#   - devices.py: a silent GPU->CPU fallback reads as slow-but-CORRECT, so KL
#     cannot see it
set -euo pipefail

# Constants read from epyc-inference-research/scripts/lib/canonical_recipe.py
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1

PPL=/mnt/raid0/llm/llama.cpp/build/bin/llama-perplexity
MODEL=/mnt/raid0/llm/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
CORPUS=${CORPUS:-/mnt/raid0/llm/ak-first-measurements/kl_corpus.txt}
CTX=${CTX:-512}
CHUNKS=${CHUNKS:-8}
PREFIX=(taskset -c 0-95 numactl --interleave=all)

mode=$1; out=$2

case "$mode" in
  dump)
    # One-time: the anchor's logits become the reference every candidate is scored on.
    "${PREFIX[@]}" "$PPL" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" \
      -t 96 -fa 1 --save-all-logits "$out" > "${out%.dat}.dump.log" 2>&1
    echo "anchor logits -> $out ($(du -h "$out" | cut -f1))"
    ;;
  score)
    # Every candidate: KL against the anchor's logits on the identical token set.
    base=$3
    /usr/bin/time -f "%e" -o "${out%.json}.elapsed" \
      "${PREFIX[@]}" "$PPL" -m "$MODEL" -f "$CORPUS" -c "$CTX" --chunks "$CHUNKS" \
      -t 96 -fa 1 --kl-divergence --kl-divergence-base "$base" > "${out%.json}.log" 2>&1
    echo "scored -> ${out%.json}.log  elapsed $(cat "${out%.json}.elapsed")s"
    ;;
  *) echo "usage: $0 {dump <out.dat> | score <out.json> <base.dat>}" >&2; exit 2 ;;
esac
