#!/bin/bash
# HS-PROF-1: BEAM extra arm — admission-schema profile vs current PREF[category] profile, on
# preference_following + instruction_following. NOT started by the build; needs an inference window
# (refuses unless WINDOW_CONFIRMED=1). Two PHASES because the judge may not be co-resident with the reader
# (docs/m12-long-context-gpu-recipe.md §6: sequential, never co-resident):
#
#   PHASE=reader   derive + build snapshots + one run_benchmark per arm; records each arm's result JSON path
#   PHASE=judge    judge smoke (5 questions) -> judge + score every arm -> table.json   (NO model swap inside)
#
# THE ONE STEP THAT STAYS A LAUNCHER STEP: between the phases the reader server is stopped and the judge server is
# started (recipe §6 line, `python scripts/benchmark/m12_launch_argv.py <judge recipe> --port P`). This script does not
# own server lifecycles (rule: kill only captured PIDs; launching a 25 GiB model is the window launcher's job), so it
# refuses to start a phase unless the port you name is owned by the pid you name. Everything else is scripted.
#
#   WINDOW_CONFIRMED=1 PHASE=reader READER_URL=http://127.0.0.1:P/v1 READER_MODEL=<id> READER_PORT=P READER_PID=<pid> \
#     OUT_DIR=/mnt/raid0/llm/tmp/hs-prof1 scripts/benchmark/run_hs_prof1_beam_arm.sh
#   WINDOW_CONFIRMED=1 PHASE=judge  JUDGE_PORT=Q JUDGE_PID=<pid> OUT_DIR=/mnt/raid0/llm/tmp/hs-prof1 \
#     scripts/benchmark/run_hs_prof1_beam_arm.sh
#
# score_beam_run is called WITHOUT --belief-measurements: that flag requires --arm to equal the rows' recorded
# context mode, which is "profile" for BOTH arms here, and beam_memory_capture has no profile-arm ladder (belief wiring is
# a recorded task in vidya-belief-substrate-program.md, not a silent omission).
# Resume: a finished arm (OUT_DIR/arm_<arm>.result.path exists and the file is there) is skipped; judge_beam_run keeps
# its own <out>.partial.jsonl. The serving binary is the STORE build in docs/m12-long-context-gpu-recipe.md (§1).
set -euo pipefail
if [[ "${WINDOW_CONFIRMED:-0}" != "1" ]]; then
  echo "refusing: needs a granted inference window (WINDOW_CONFIRMED=1)" >&2; exit 3
fi
: "${PHASE:?PHASE=reader|judge}" "${OUT_DIR:?}"
PY="${PY:-/workspace/repos/epyc-orchestrator/.venv/bin/python}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/raid0/llm/epyc-inference-research/benchmarks/results}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma-4-26B-A4B-it-ORIG-Q8_0}"
ARMS=(pref_current admission)
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"

prove_port() {  # <port> <pid> : the port must be a non-production port owned by the pid the caller started
  local port="$1" pid="$2" owner
  case "$port" in 8000|8070|8071|8072|8080|8081|8082|8083|8090|8100|8180|8181|8182) echo "refusing: $port is a production stack port" >&2; exit 4;; esac
  owner="$(ss -H -ltnp "sport = :$port" | sed -nE 's/.*pid=([0-9]+).*/\1/p' | head -1)"
  [[ -n "$owner" && "$owner" == "$pid" ]] || { echo "refusing: port $port owner is '${owner:-none}', not pid $pid (a foreign server may answer)" >&2; exit 4; }
}
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUT_DIR/run.log" >&2; }

if [[ "$PHASE" == "reader" ]]; then
  : "${READER_URL:?}" "${READER_MODEL:?}" "${READER_PORT:?}" "${READER_PID:?pid of the private reader server}"
  prove_port "$READER_PORT" "$READER_PID"
  cd "$here"
  [[ -s "$OUT_DIR/derived.json" ]] || "$PY" hs_prof1_profile_arms.py derive --base-url "$READER_URL" --model "$READER_MODEL" --out "$OUT_DIR/derived.json"
  "$PY" hs_prof1_profile_arms.py build --derived "$OUT_DIR/derived.json" --out-dir "$OUT_DIR"
  for arm in "${ARMS[@]}"; do
    if [[ -s "$OUT_DIR/arm_${arm}.result.path" && -f "$(cat "$OUT_DIR/arm_${arm}.result.path")" ]]; then log "arm $arm already done, skipping"; continue; fi
    prove_port "$READER_PORT" "$READER_PID"            # again per arm: the server must still be ours
    BEAM_CONTEXT_MODE=profile BEAM_PROFILE_SNAPSHOTS="$OUT_DIR/snapshots_${arm}.json" BEAM_PROFILE_ARM="$arm" \
    BEAM_ABILITIES=preference_following,instruction_following \
    "$PY" run_benchmark.py --suite beam --server-mode --existing-server-port "$READER_PORT" \
      --skip-moe-reduction --skip-speed-tests --new-run 2>&1 | tee "$OUT_DIR/arm_${arm}.log"
    run_id="$(sed -nE 's/^New run: (.+)$/\1/p' "$OUT_DIR/arm_${arm}.log" | head -1)"
    [[ -n "$run_id" ]] || { echo "could not read the run id from arm_${arm}.log" >&2; exit 5; }
    result="$(ls -1 "$RESULTS_ROOT/runs/$run_id"/*.json 2>/dev/null | grep -v manifest.json | head -1 || true)"
    [[ -n "$result" ]] || { echo "no result JSON under $RESULTS_ROOT/runs/$run_id" >&2; exit 5; }
    "$PY" - "$result" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); r = d.get("results", d)
if not (isinstance(r, dict) and r.get("beam")):
    sys.exit("result has no results['beam'] rows: " + sys.argv[1])
print("beam rows:", len(r["beam"]))
PYEOF
    printf '%s\n' "$result" > "$OUT_DIR/arm_${arm}.result.path"
    log "arm $arm -> $result"
  done
  log "reader phase done. NEXT (launcher step): stop the reader (captured pid), start the judge, then PHASE=judge."
elif [[ "$PHASE" == "judge" ]]; then
  : "${JUDGE_PORT:?}" "${JUDGE_PID:?pid of the private judge server}"
  prove_port "$JUDGE_PORT" "$JUDGE_PID"
  JURL="http://127.0.0.1:${JUDGE_PORT}"
  first="${ARMS[0]}"
  [[ -s "$OUT_DIR/arm_${first}.result.path" ]] || { echo "no reader result recorded; run PHASE=reader first" >&2; exit 5; }
  cd "$here"
  # judge smoke on a 5-question slice (recipe §6): unjudged empty, every verdict parses, else stop before the long judge pass
  "$PY" - "$(cat "$OUT_DIR/arm_${first}.result.path")" "$OUT_DIR/smoke_slice.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); r = d.get("results", d)
keep = dict(list(r["beam"].items())[:5])
r2 = dict(r); r2["beam"] = keep
json.dump({**d, "results": r2} if "results" in d else r2, open(sys.argv[2], "w"))
PYEOF
  "$PY" judge_beam_run.py "$OUT_DIR/smoke_slice.json" --out-json "$OUT_DIR/judged_smoke.json" --judge-model "$JUDGE_MODEL" --judge-url "$JURL"
  "$PY" - "$OUT_DIR/judged_smoke.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("unjudged"):
    sys.exit(f"judge smoke FAILED: unjudged={d['unjudged']}")
print("judge smoke ok")
PYEOF
  for arm in "${ARMS[@]}"; do
    prove_port "$JUDGE_PORT" "$JUDGE_PID"
    "$PY" judge_beam_run.py "$(cat "$OUT_DIR/arm_${arm}.result.path")" --out-json "$OUT_DIR/judged_${arm}.json" --judge-model "$JUDGE_MODEL" --judge-url "$JURL"
    "$PY" score_beam_run.py "$OUT_DIR/judged_${arm}.json" --out-json "$OUT_DIR/scored_${arm}.json" --check-dataset
  done
  "$PY" hs_prof1_profile_arms.py table --out-dir "$OUT_DIR" --out "$OUT_DIR/table.json"
  log "judge phase done: $OUT_DIR/table.json"
else
  echo "PHASE must be reader or judge" >&2; exit 2
fi
