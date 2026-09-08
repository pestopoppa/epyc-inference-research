#!/bin/bash
# INF-70 PROD-1 — the ONLY sanctioned launcher for Qwen3.8-Flash-Next on the CPU path.
#
# INTENDED REPO PATH: epyc-inference-research/scripts/benchmark/serve_qwen38_flash_next.sh
#
# It RESTATES NO CONSTANTS. Every flag, env var and path comes out of
# scripts/lib/qwen38_flash_next_recipe.py as JSON. If you find yourself typing a
# llama-server flag into this file, stop: put it in the recipe module instead.
#
# The MTP head is not optional (OP-35). --plain exists only to produce the CONTROL ARM.
#
# usage: serve_qwen38_flash_next.sh [--bindir DIR] [--model GGUF] [--port N]
#                                   [--host H] [--plain] [--dry-run] [--skip-digests]
set -euo pipefail

RECIPE_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/qwen38_flash_next_recipe.py"
[ -f "$RECIPE_LIB" ] || { echo "FATAL: recipe module not found at $RECIPE_LIB" >&2; exit 78; }

DRYRUN=0
PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRYRUN=1; shift;;
    --bindir|--model|--port|--host) PASS+=("$1" "$2"); shift 2;;
    --plain|--skip-digests) PASS+=("$1"); shift;;
    *) echo "unknown argument: $1" >&2; exit 64;;
  esac
done
# --skip-digests is a preflight-only option; strip it from the emit call.
EMIT_ARGS=(); for a in "${PASS[@]}"; do [ "$a" = "--skip-digests" ] || EMIT_ARGS+=("$a"); done

# 1. PREFLIGHT. Costs seconds, needs no region lock, no model load and no GPU. It
#    dry-runs every flag form against the real binary — the check that would have
#    saved the seven MTP arms SYNC-10 lost to `--fa 1`.
python3 "$RECIPE_LIB" preflight "${PASS[@]}"

# 2. Ask the recipe for the command and the environment. Nothing is reconstructed here.
PLAN=$(python3 "$RECIPE_LIB" emit-serve-command "${EMIT_ARGS[@]}")
printf '%s' "$PLAN" | python3 -c 'import json,sys
d=json.load(sys.stdin)
print("recipe_sha256=" + d["recipe_sha256"] + " mtp=" + str(d["mtp"]))'

mapfile -t CMD   < <(printf '%s' "$PLAN" | python3 -c 'import json,sys;[print(c) for c in json.load(sys.stdin)["cmd"]]')
mapfile -t ENVKV < <(printf '%s' "$PLAN" | python3 -c 'import json,sys;[print(f"{k}={v}") for k,v in json.load(sys.stdin)["env"].items()]')
BINDIR=$(printf '%s' "$PLAN" | python3 -c 'import json,sys;print(json.load(sys.stdin)["ld_library_path_prefix"])')

# 3. Three ggml generations live on this host; a binary that inherits another tree's
#    ggml runs silently wrong. Prepend the binary's own directory, then PROVE residency
#    after launch by reading the libggml-cpu.so path out of /proc/PID/maps.
export LD_LIBRARY_PATH="$BINDIR:${LD_LIBRARY_PATH:-}"

if [ "$DRYRUN" = 1 ]; then
  printf '%s\n' "${ENVKV[@]}" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

cat >&2 <<'NOTE'
NOTE: this launcher does NOT take the bench region lock and does NOT evict the page
      cache. To reproduce a HEADLINE number, wrap it in
        region-lock run --cpu-list 0-95 --role bench -- ...
      and satisfy every entry of the recipe's PRECONDITIONS dict first (eviction,
      load gate, linkage proof, placement proof, env readback). A number measured
      without those is not comparable to the recipe's headline.
NOTE

exec env "${ENVKV[@]}" "${CMD[@]}"
