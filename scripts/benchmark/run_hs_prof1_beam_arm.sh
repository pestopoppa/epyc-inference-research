#!/bin/bash
# HS-PROF-1: BEAM extra arm — admission-schema profile vs current PREF[category] profile, on
# preference_following + instruction_following. NOT started by the build; needs an inference window
# (refuses unless WINDOW_CONFIRMED=1). Reader served on an OpenAI-compatible endpoint, judge per
# docs/m12-long-context-gpu-recipe.md §6.
#
#   WINDOW_CONFIRMED=1 READER_URL=http://127.0.0.1:PORT/v1 READER_MODEL=<id> READER_PORT=PORT \
#     OUT_DIR=/mnt/raid0/llm/tmp/hs-prof1 scripts/benchmark/run_hs_prof1_beam_arm.sh
set -euo pipefail
if [[ "${WINDOW_CONFIRMED:-0}" != "1" ]]; then
  echo "refusing: needs a granted inference window (WINDOW_CONFIRMED=1)" >&2; exit 3
fi
: "${READER_URL:?}" "${READER_MODEL:?}" "${READER_PORT:?}" "${OUT_DIR:?}"
PY="${PY:-/workspace/repos/epyc-orchestrator/.venv/bin/python}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"
cd "$here"
# 1 deriver pass (model call), 2 offline snapshot build for both arms
"$PY" hs_prof1_profile_arms.py derive --base-url "$READER_URL" --model "$READER_MODEL" --out "$OUT_DIR/derived.json"
"$PY" hs_prof1_profile_arms.py build --derived "$OUT_DIR/derived.json" --out-dir "$OUT_DIR"
# 3 one run_benchmark invocation per arm (same flags as the M-12b arms)
for arm in pref_current admission; do
  BEAM_CONTEXT_MODE=profile BEAM_PROFILE_SNAPSHOTS="$OUT_DIR/snapshots_${arm}.json" BEAM_PROFILE_ARM="$arm" \
  BEAM_ABILITIES=preference_following,instruction_following \
  "$PY" run_benchmark.py --suite beam --server-mode --existing-server-port "$READER_PORT" \
    --skip-moe-reduction --skip-speed-tests --new-run
done
echo "next: launch the judge, then judge_beam_run.py + score_beam_run.py per arm (recipe §7 steps 6-7); file the table"
