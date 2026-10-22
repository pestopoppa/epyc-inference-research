#!/bin/bash
# HS-PROF-1: BEAM extra arm — admission-schema profile vs current PREF[category] profile, on
# preference_following + instruction_following. NOT started by the build; needs an inference window
# (refuses unless WINDOW_CONFIRMED=1). Two PHASES because the judge may not be co-resident with the reader
# (docs/m12-long-context-gpu-recipe.md §6: sequential, never co-resident):
#
#   PHASE=reader   derive + build snapshots + one run_benchmark per arm; records each arm's result JSON path
#   PHASE=preflight no server/model/judge call: pyarrow + orchestrator imports + BEAM extract_all + tree-resolution check (run it BEFORE the GPU grant)
#   PHASE=judge    (JUDGE_BACKEND=served, default) judge smoke (5 questions) -> judge + score every arm -> table.json   (NO model swap inside)
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
# JUDGE_BACKEND=cloud (operator 2026-10-22: Claude via the claude_cli cloud judge, subscription CLI auth; gemma ORIG retired):
# PHASE=judge needs NO JUDGE_PORT/JUDGE_PID and no server proof (no GPU, no local model). Transport = orchestrator
# typed_decisions/cloud_judges.py, registry orchestration/cloud_judges.yaml entry $CLOUD_JUDGE (default sonnet-low). No API key
# anywhere. judge_beam_run prints the call estimate, then makes ONE preflight `claude -p` call; if `claude` is missing or not
# logged in it prints "BLOCKED: ..." and exits 5. The 5-question smoke still runs first. JUDGE_WORKERS (default 2) = questions
# judged concurrently.
#   WINDOW_CONFIRMED=1 PHASE=judge JUDGE_BACKEND=cloud OUT_DIR=... scripts/benchmark/run_hs_prof1_beam_arm.sh
#
# score_beam_run is called WITHOUT --belief-measurements: that flag requires --arm to equal the rows' recorded
# context mode, which is "profile" for BOTH arms here. The belief rows for this experiment are written instead by
# `hs_prof1_profile_arms.py table --belief-measurements` (VB-HSPROF1-WRITE: per ability x arm means + paired deltas,
# judge identity in every claim) through epyc-root's scripts/vidya/adapters/hs_prof1_profile_arms.py. It needs
# EPYC_ROOT=<epyc-root checkout>; the judge phase preflights it BEFORE any judge call (exit 5 if unresolvable).
# HSPROF1_BELIEF=0 skips the capture explicitly (logged). READER_RECIPE (optional) is recorded with its sha256.
# Resume: a finished arm (OUT_DIR/arm_<arm>.result.path exists and the file is there) is skipped; judge_beam_run keeps
# its own <out>.partial.jsonl. The serving binary is the STORE build in docs/m12-long-context-gpu-recipe.md (§1).
set -euo pipefail
if [[ "${WINDOW_CONFIRMED:-0}" != "1" ]]; then
  echo "refusing: needs a granted inference window (WINDOW_CONFIRMED=1)" >&2; exit 3
fi
: "${PHASE:?PHASE=reader|judge|preflight}" "${OUT_DIR:?}"
# One interpreter for every step. The orchestrator venv has NO pyarrow (BEAM parquet load refuses without it); the research venv has
# pyarrow AND imports the orchestrator's src.user_modeling / cloud_judges fine (pure python, loaded off EPYC_ORCH_ROOT). INC 2026-10-22.
PY="${PY:-/mnt/raid0/llm/epyc-inference-research/.venv/bin/python}"
export EPYC_ORCH_ROOT="${EPYC_ORCH_ROOT:-/workspace/repos/epyc-orchestrator}"
RESULTS_ROOT="${RESULTS_ROOT:-/mnt/raid0/llm/epyc-inference-research/benchmarks/results}"
JUDGE_BACKEND="${JUDGE_BACKEND:-served}"
CLOUD_JUDGE="${CLOUD_JUDGE:-sonnet-low}"
JUDGE_WORKERS="${JUDGE_WORKERS:-2}"
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
  # a derived.json that is empty/unparsable is NOT reusable (2026-10-22: a 20/20-empty file was kept and re-fed): validate before skipping derive
  if [[ -s "$OUT_DIR/derived.json" ]] && ! "$PY" -c 'import json,sys,hs_prof1_profile_arms as h; h.check_derived(json.load(open(sys.argv[1])))' "$OUT_DIR/derived.json"; then
    echo "stale/empty derived.json: moving aside and re-deriving" >&2; mv "$OUT_DIR/derived.json" "$OUT_DIR/derived.bad.$(date -u +%s).json"
  fi
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
elif [[ "$PHASE" == "preflight" ]]; then
  # No server, no model, no judge call: prove the interpreter + code tree BEFORE a GPU grant / server launch is spent.
  # Checks: pyarrow importable, orchestrator imports (_orch), BEAM parquet loads (extract_all, data only), the modules resolve
  # to THIS tree ($here), run_benchmark imports, and (if EPYC_ROOT is set) the belief writer resolves.
  cd "$here"
  "$PY" - "$here" "${EPYC_ROOT:-}" <<'PYEOF' || { echo "BLOCKED: preflight failed under PY=$PY (see above); no server started" >&2; exit 5; }
import os, sys
here = os.path.realpath(sys.argv[1]); epyc_root = sys.argv[2]
sys.path.insert(0, here); sys.path.insert(1, os.path.dirname(here))  # scripts/ for lib.executor
try:
    import pyarrow
except ImportError as e:
    sys.exit(f"preflight: interpreter {sys.executable} has no pyarrow ({e})")
import hs_prof1_profile_arms as h, long_context_adapters as L, judge_beam_run, score_beam_run
for m in (h, L, judge_beam_run, score_beam_run):
    if os.path.dirname(os.path.realpath(m.__file__)) != here:
        sys.exit(f"preflight: {m.__name__} resolved to {m.__file__}, not the tree {here}")
h._orch()
ad = L.BEAMAdapter(data_dir=None, split="100K", context_mode="full")
n = len(ad.extract_all())
if n <= 0 or not ad._transcripts:
    sys.exit("preflight: BEAM extract_all returned nothing")
if epyc_root:
    import belief_capture; belief_capture.preflight("hs_prof1_profile_arms")
# thinking flag: the deriver request body AND the BEAM answering payload (via the real ServerManager chat path, HTTP mocked) must carry
# chat_template_kwargs.enable_thinking == False, and the BEAM suite must pin enable_thinking False. No network, no server.
body = h.build_derive_request("m", "p")
if body.get("chat_template_kwargs") != {"enable_thinking": False}:
    sys.exit(f"preflight: deriver request lacks enable_thinking=false: {body}")
if L.BEAMAdapter.inference_params.get("enable_thinking") is not False:
    sys.exit("preflight: BEAM inference_params does not pin enable_thinking=False")
import json as _json
from unittest import mock
from lib.executor import ServerManager
sm = ServerManager.__new__(ServerManager); sm.port = 1; sm.mmproj_path = None; sm.use_chat_api = False
class _R:  # canned chat reply
    status_code = 200; text = ""
    def json(self): return {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}], "usage": {}, "timings": {}}
sess = mock.Mock(); sess.post.return_value = _R()
sm._get_http_session = lambda: sess
sm.run_inference(prompt="q", max_tokens=8, temperature=0.0, timeout=5, enable_thinking=L.BEAMAdapter.inference_params["enable_thinking"])
url = sess.post.call_args.args[0]; sent = sess.post.call_args.kwargs["json"]
if not url.endswith("/v1/chat/completions") or sent.get("chat_template_kwargs") != {"enable_thinking": False}:
    sys.exit(f"preflight: BEAM answer request not chat+enable_thinking=false: {url} {sent}")
import run_benchmark as _rb
if _rb.canonical_preflight_skip_reason(type("A", (), {"skip_preflight": False, "existing_server_port": 18977})()) is None:
    sys.exit("preflight: run_benchmark would run the CPU canonical preflight with --existing-server-port")
print(f"preflight ok: py={sys.executable} pyarrow={pyarrow.__version__} questions={n} conversations={len(ad._transcripts)} tree={here}")
PYEOF
  EPYC_ORCH_ROOT="$EPYC_ORCH_ROOT" "$PY" -c 'import sys; sys.argv=["run_benchmark.py","--help"]; sys.path.insert(0,"."); import run_benchmark' >/dev/null 2>&1 \
    || { echo "BLOCKED: run_benchmark.py not importable under PY=$PY" >&2; exit 5; }
elif [[ "$PHASE" == "judge" ]]; then
  case "$JUDGE_BACKEND" in
    served)
      : "${JUDGE_PORT:?}" "${JUDGE_PID:?pid of the private judge server}"
      prove_port "$JUDGE_PORT" "$JUDGE_PID"
      JARGS=(--judge-model "$JUDGE_MODEL" --judge-url "http://127.0.0.1:${JUDGE_PORT}")
      proof() { prove_port "$JUDGE_PORT" "$JUDGE_PID"; } ;;
    cloud)
      # --judge-model left to default (<registry name>:<resolved model>) unless JUDGE_MODEL_ID pins another allow-listed id
      JARGS=(--judge-backend cloud --cloud-judge "$CLOUD_JUDGE" --workers "$JUDGE_WORKERS")
      [[ -z "${JUDGE_MODEL_ID:-}" ]] || JARGS+=(--judge-model-id "$JUDGE_MODEL_ID")
      proof() { :; } ;;   # no server, no port, no pid
    *) echo "JUDGE_BACKEND must be served or cloud" >&2; exit 2 ;;
  esac
  first="${ARMS[0]}"
  [[ -s "$OUT_DIR/arm_${first}.result.path" ]] || { echo "no reader result recorded; run PHASE=reader first" >&2; exit 5; }
  cd "$here"
  TARGS=()
  if [[ "${HSPROF1_BELIEF:-1}" == "1" ]]; then
    # VB-HSPROF1-WRITE: resolve the belief writer before spending any judge call
    "$PY" -c 'import belief_capture; belief_capture.preflight("hs_prof1_profile_arms")' \
      || { echo "BLOCKED: belief capture unresolvable (set EPYC_ROOT, or HSPROF1_BELIEF=0 to skip it explicitly)" >&2; exit 5; }
    TARGS+=(--belief-measurements)
    [[ -z "${READER_RECIPE:-}" ]] || TARGS+=(--reader-recipe "$READER_RECIPE")
  else
    log "HSPROF1_BELIEF=0: belief capture SKIPPED by request (no belief rows for this table)"
  fi
  # judge smoke on a 5-question slice (recipe §6): unjudged empty, every verdict parses, else stop before the long judge pass
  "$PY" - "$(cat "$OUT_DIR/arm_${first}.result.path")" "$OUT_DIR/smoke_slice.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1])); r = d.get("results", d)
keep = dict(list(r["beam"].items())[:5])
r2 = dict(r); r2["beam"] = keep
json.dump({**d, "results": r2} if "results" in d else r2, open(sys.argv[2], "w"))
PYEOF
  "$PY" judge_beam_run.py "$OUT_DIR/smoke_slice.json" --out-json "$OUT_DIR/judged_smoke.json" "${JARGS[@]}"
  "$PY" - "$OUT_DIR/judged_smoke.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("unjudged"):
    sys.exit(f"judge smoke FAILED: unjudged={d['unjudged']}")
print("judge smoke ok")
PYEOF
  for arm in "${ARMS[@]}"; do
    proof
    "$PY" judge_beam_run.py "$(cat "$OUT_DIR/arm_${arm}.result.path")" --out-json "$OUT_DIR/judged_${arm}.json" "${JARGS[@]}"
    "$PY" score_beam_run.py "$OUT_DIR/judged_${arm}.json" --out-json "$OUT_DIR/scored_${arm}.json" --check-dataset
  done
  "$PY" hs_prof1_profile_arms.py table --out-dir "$OUT_DIR" --out "$OUT_DIR/table.json" "${TARGS[@]}"
  log "judge phase done: $OUT_DIR/table.json"
else
  echo "PHASE must be reader, judge or preflight" >&2; exit 2
fi
