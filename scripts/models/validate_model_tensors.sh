#!/bin/bash
# Acquisition-time GGUF tensor validation. Run ONCE when a model lands on disk.
#
#   bash scripts/models/validate_model_tensors.sh /path/to/model.gguf
#   bash scripts/models/validate_model_tensors.sh --check /path/to/model.gguf   # has it been validated?
#   bash scripts/models/validate_model_tensors.sh --list                        # what has been validated
#
# WHY AT ACQUISITION AND NOT AT SERVE (operator decision, 2026-08-03)
#   `--check-tensors` reads every tensor linearly. On an mmap'd model that forces
#   FIRST-TOUCH OF EVERY PAGE, which adds a full model read to every server start
#   and perturbs NUMA first-touch placement -- a hazard this project has already
#   been bitten by. Running it once at acquisition gets the same detection with
#   none of that: the serving path is untouched.
#
# WHAT IT ACTUALLY CATCHES
#   ggml_validate_row_data checks NaN/Inf in fp16 scale fields for the quant types
#   we serve (Q4_K, Q8_0, IQ2_XXS, BF16, F16) and the reserved 0xFF exponent for
#   MXFP4. So this is a CORRUPT-DOWNLOAD DETECTOR first, and the E8M0 gate second.
#   A corrupted scale in a 38 GB download produces garbage output, not a crash --
#   which is precisely the failure mode that survives without a check like this.
#
# RESOURCE NOTE
#   Validation loads the model. This is a deliberate, scheduled action; it is not
#   safe to fire during a busy serving window, and the script refuses if free RAM
#   cannot hold the file.

set -euo pipefail

LLAMA="${LLAMA_CPP_ROOT:-/mnt/raid0/llm/llama.cpp}"
RECEIPTS="${MODEL_VALIDATION_RECEIPTS:-/mnt/raid0/llm/models/.tensor-validation}"
# BUILD SELECTION IS NOT INCIDENTAL. Of the 12 builds on this host, 8 are BROKEN
# (stale link, `undefined symbol: llama_apply_adapter_cvec`); an earlier version of
# this script took the first one that existed and reported its linkage error as
# "this model is invalid" -- a verdict about the model from a fault in the tool.
# So: require a build that RUNS, prefer the ratified production version, and prefer
# non-HIP because validation is CPU-side and this host serves from the GPU.
PROD_VERSION="${LLAMA_PROD_VERSION:-10107}"
BUILD=""; BUILD_VER=""
for cand in "$LLAMA"/build*/bin/llama-cli; do
  [[ -x "$cand" ]] || continue
  [[ -f "$(dirname "$cand")/libllama.so" ]] || continue
  ver=$(timeout 20 "$cand" --version 2>&1 | head -1) || continue
  [[ "$ver" == *"version:"* ]] || continue
  if [[ "$ver" == *"$PROD_VERSION"* && "$cand" != *build-hip* ]]; then
    BUILD="$(dirname "$cand")"; BUILD_VER="$ver"; break
  fi
  [[ -z "$BUILD" ]] && BUILD="$(dirname "$cand")" && BUILD_VER="$ver"
done

HARNESS="$(dirname "${BASH_SOURCE[0]}")/validate_tensors_harness"

usage() { sed -n '2,26p' "$0"; exit 0; }

mkdir -p "$RECEIPTS"

receipt_for() {  # model path -> receipt path, keyed by realpath hash so moves are visible
  local rp; rp="$(readlink -f "$1")"
  printf '%s/%s.json' "$RECEIPTS" "$(printf '%s' "$rp" | sha256sum | cut -c1-16)"
}

case "${1:-}" in
  -h|--help|"") usage ;;
  --list)
    shopt -s nullglob
    n=0
    for r in "$RECEIPTS"/*.json; do
      python3 -c "
import json,sys
d=json.load(open('$r'))
print(f\"  {d['result']:<8} {d['validated_utc']}  {d['model']}\")"
      n=$((n+1))
    done
    [[ $n -eq 0 ]] && echo "  (no models validated yet)"
    exit 0 ;;
  --check)
    MODEL="${2:?--check needs a model path}"
    R="$(receipt_for "$MODEL")"
    if [[ -f "$R" ]]; then
      res=$(python3 -c "import json;print(json.load(open('$R'))['result'])")
      echo "$res  $MODEL"
      [[ "$res" == "PASS" ]] && exit 0 || exit 1
    fi
    echo "NOT-VALIDATED  $MODEL"
    exit 2 ;;
esac

MODEL="$1"
[[ -f "$MODEL" ]] || { echo "REFUSING: no such file: $MODEL" >&2; exit 1; }
if [[ -z "$BUILD" ]]; then
  echo "REFUSING: no WORKING llama.cpp build with libllama.so found under $LLAMA/build*/bin." >&2
  echo "Refusing rather than emitting a verdict from a binary that cannot run." >&2
  exit 1
fi
echo "   build  $BUILD"
echo "   $BUILD_VER"

if [[ ! -x "$HARNESS" ]] || [[ "$HARNESS.c" -nt "$HARNESS" ]]; then
  echo "   building harness"
  cc -O2 -I"$LLAMA/include" -I"$LLAMA/ggml/include" -o "$HARNESS" "$HARNESS.c" \
     -L"$BUILD" -lllama -Wl,-rpath,"$BUILD" || { echo "REFUSING: harness build failed" >&2; exit 1; }
fi

SIZE_KB=$(du -k "$MODEL" | cut -f1)
FREE_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
echo "== validating $(basename "$MODEL")"
echo "   size $((SIZE_KB/1024/1024)) GiB · free RAM $((FREE_KB/1024/1024)) GiB"
if (( FREE_KB < SIZE_KB + 4194304 )); then
  echo "REFUSING: not enough free RAM to load this model plus 4 GiB headroom." >&2
  echo "Validation loads the model; running it against a busy host would compete with serving." >&2
  exit 3
fi

echo "   loading with check_tensors=true (reads every tensor; one full pass over the file)"
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT
set +e
# The harness calls llama_model_load_from_file directly. llama-cli was tried first
# and is unusable here: with a non-TTY stdin it enters an interactive loop and emits
# "> " forever (observed: 312 million lines, 895 MB, with -no-cnv AND stdin from
# /dev/null). The loader API gives the boolean we want and nothing else.
timeout "${VALIDATE_TIMEOUT_S:-3600}" "$HARNESS" "$MODEL" >"$LOG" 2>&1
RC=$?
set -e

# THREE STATES, NOT TWO. A tool that cannot run has not found a bad model, and
# reporting FAIL for that would send someone to re-download a good file. FAIL is
# emitted ONLY on a positive validation rejection; every other non-zero exit is
# ERROR, which is inconclusive and blocks nothing.
if grep -q '"result": "PASS"' "$LOG"; then
  RESULT=PASS
elif grep -q '"result": "FAIL"' "$LOG" || grep -qiE 'invalid data|found invalid' "$LOG"; then
  RESULT=FAIL
else
  RESULT=ERROR    # timeout, crash, missing lib: says NOTHING about the model
fi

SHA=$(sha256sum "$MODEL" | cut -d' ' -f1)
python3 - "$(receipt_for "$MODEL")" "$MODEL" "$SHA" "$RESULT" "$RC" <<'PY'
import json, sys, datetime, subprocess
path, model, sha, result, rc = sys.argv[1:6]
json.dump({
    "model": model,
    "sha256": sha,
    "result": result,
    "exit_code": int(rc),
    "validated_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "what_was_checked": ("ggml_validate_row_data: NaN/Inf in fp16 scale fields for the served quant "
                         "types, and the reserved 0xFF E8M0 exponent for MXFP4"),
    "why_not_at_serve": ("--check-tensors forces first-touch of every page on mmap'd models, adding a "
                         "full read to every start and perturbing NUMA placement. Operator decision "
                         "2026-08-03: validate once at acquisition instead."),
}, open(path, "w"), indent=2)
PY

if [[ "$RESULT" == "FAIL" ]]; then
  echo
  echo "FAIL — tensor validation REJECTED this model:"
  grep -iE 'invalid data|found invalid' "$LOG" | head -8 | sed 's/^/    /'
  echo
  echo "This is what the check exists for: a corrupted scale would otherwise produce"
  echo "garbage output rather than an error. Re-download before serving it."
  exit 4
fi

if [[ "$RESULT" == "ERROR" ]]; then
  echo
  echo "ERROR — the validator could not run to completion (exit $RC). This says NOTHING"
  echo "about the model; do not re-download on the strength of it. Last log lines:"
  tail -5 "$LOG" | sed 's/^/    /'
  exit 5
fi

echo "   PASS — receipt at $(receipt_for "$MODEL")"
