#!/bin/bash
# CME-7: run the CL-Bench smoke (exploitable_poker, quick_test: 5 hands x 3 runs) through the
# epyc_local system against a PRIVATE local OpenAI-compatible llama-server. NOT started by the build.
#
# Needs an inference window: refuses unless WINDOW_CONFIRMED=1. It also refuses unless the endpoint's port
# is owned by the pid you name (a foreign server answers /models too) and is not a production port.
#
# CME-5 SIZING (decided 2026-10-21, queue-builders-3; see handoffs/active/conversational-memory-eval-instrument.md):
#   * 3 runs execute CONCURRENTLY, each an append-only history -> the server needs >= 3 slots, one per run.
#   * per-run cap --system.max-context-tokens 120000 (default) -> >= 131072 positions per slot.
#   * NO new recipe: use the existing eval recipe qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4 (4 slots x 196608 per slot,
#     --ctx-checkpoints, --cache-ram, prompt cache on, --no-kv-unified, taskset 184-191, MI210), launched from
#       python scripts/benchmark/m12_launch_argv.py artifacts/serving-recipes/eval/qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4.json --port <PRIVATE>
#     on a PRIVATE port (not 8199: that is the operator's replay server). The smoke's real history is far shorter than the cap
#     (5 hands); the cap is chosen so the truncation path never fires and a later full schedule reuses the same server recipe.
#   * acceptance: all 3 runs x 5 hands complete, and cache_hit_report.py hit_ratio_reusable >= 0.95 (its docstring defines
#     why the gate is over prefix-reusable tokens, not raw cached/input).
#
#   WINDOW_CONFIRMED=1 CLBENCH_EPYC_BASE_URL=http://127.0.0.1:PORT/v1 CLBENCH_SERVER_PID=<pid of that llama-server> \
#     EPYC_MODEL=<served model id> scripts/benchmark/run_clbench_epyc_local.sh
#
# Defaults: CLBENCH_REPO=/mnt/raid0/llm/tmp/clbench-epyc-repo (clone @ 5f8c50e, epyc_local installed),
#           CLBENCH_VENV=/mnt/raid0/llm/tmp/clbench-venv (python 3.13, litellm/typer/texasholdem; `clbench` console script).
# NOTE: clbench's argparse takes HYPHENATED system flags (--system.base-url, --system.max-context-tokens);
#       the underscore spellings are rejected ("unrecognized arguments") -- verified 2026-10-21.
set -euo pipefail

if [[ "${WINDOW_CONFIRMED:-0}" != "1" ]]; then
  echo "refusing: this run needs a granted inference window (set WINDOW_CONFIRMED=1)" >&2
  exit 3
fi
CLBENCH_REPO="${CLBENCH_REPO:-/mnt/raid0/llm/tmp/clbench-epyc-repo}"
CLBENCH_VENV="${CLBENCH_VENV:-/mnt/raid0/llm/tmp/clbench-venv}"
: "${CLBENCH_EPYC_BASE_URL:?set CLBENCH_EPYC_BASE_URL to the served endpoint, e.g. http://127.0.0.1:PORT/v1}"
: "${CLBENCH_SERVER_PID:?set CLBENCH_SERVER_PID to the pid of the private llama-server you started}"
: "${EPYC_MODEL:?set EPYC_MODEL to the served model id}"
TASK="${CLBENCH_TASK:-exploitable_poker}"
SCHEDULE="${CLBENCH_SCHEDULE:-quick_test}"
MAX_CTX="${CLBENCH_MAX_CONTEXT_TOKENS:-120000}"
OUT_ROOT="${CLBENCH_OUT_ROOT:-/mnt/raid0/llm/tmp/clbench-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ -x "$CLBENCH_VENV/bin/clbench" ]] || { echo "no clbench in $CLBENCH_VENV (build the venv: python3.13 -m venv + pip install -e --no-deps the repo)" >&2; exit 5; }
[[ -d "$CLBENCH_REPO/src" ]] || { echo "CLBENCH_REPO $CLBENCH_REPO is not a CL-Bench clone" >&2; exit 5; }

# Port-ownership proof: the endpoint's port must be owned by the pid the caller started, and not be a production port.
port="$(sed -E 's#^https?://[^:/]+:([0-9]+).*#\1#' <<<"$CLBENCH_EPYC_BASE_URL")"
[[ "$port" =~ ^[0-9]+$ ]] || { echo "cannot parse port from $CLBENCH_EPYC_BASE_URL" >&2; exit 4; }
case "$port" in 8000|8070|8071|8072|8080|8081|8082|8083|8090|8100|8180|8181|8182) echo "refusing: $port is a production stack port" >&2; exit 4;; esac
owner="$(ss -H -ltnp "sport = :$port" | sed -nE 's/.*pid=([0-9]+).*/\1/p' | head -1)"
[[ "$owner" == "$CLBENCH_SERVER_PID" ]] || { echo "refusing: port $port owner is '${owner:-none}', not CLBENCH_SERVER_PID=$CLBENCH_SERVER_PID" >&2; exit 4; }
curl -fsS --max-time 10 "${CLBENCH_EPYC_BASE_URL%/}/models" >/dev/null \
  || { echo "endpoint ${CLBENCH_EPYC_BASE_URL} not answering /models" >&2; exit 4; }

"$here/clbench_epyc_local/install_into_clbench.sh" "$CLBENCH_REPO"
mkdir -p "$OUT_ROOT"
# CME-7T: failed / empty / runaway raw replies land next to cache_hit_report.json (system.py honours this env)
export CLBENCH_EPYC_RAW_LOG="$(cd "$OUT_ROOT" && pwd)/raw_replies.jsonl"
cd "$CLBENCH_REPO"
before="$(ls -d results/"$TASK"/traces/*/ 2>/dev/null | sort || true)"
set +e
"$CLBENCH_VENV/bin/clbench" run "$TASK" --schedule "$SCHEDULE" --system epyc_local \
  --system.model "$EPYC_MODEL" --system.base-url "$CLBENCH_EPYC_BASE_URL" \
  --system.max-context-tokens "$MAX_CTX" --no-live-dashboard 2>&1 | tee "$OUT_ROOT/clbench_run.log"
rc=${PIPESTATUS[0]}
set -e
traces="$(comm -13 <(printf '%s\n' "$before") <(ls -d results/"$TASK"/traces/*/ | sort) | tail -1)"
echo "run rc=$rc traces=$traces" | tee "$OUT_ROOT/status.txt"
[[ "$rc" == "0" && -n "$traces" ]] || exit "${rc:-1}"
python3 -I "$here/clbench_epyc_local/cache_hit_report.py" "$CLBENCH_REPO/$traces" --expect-runs 3 --expect-hands 5 \
  --out "$OUT_ROOT/cache_hit_report.json" && echo "cache gate PASS" || echo "cache gate rc=$? (see $OUT_ROOT/cache_hit_report.json)"
