#!/bin/bash
set -euo pipefail
# dflash2_capability_smoke.sh <build-dir> <out-dir>
#
# Verifies a champion build's DFlash2 capability end-to-end, fail-closed:
#   1. llama-server loads target 27B + DFlash2 drafter head under --spec-type draft-dflash
#      (the head is a companion artifact — it can NEVER be smoked as a standalone model;
#      that was the boundary-20260901 step5 mistake).
#   2. A pinned greedy completion produces tokens.
#   3. Draft acceptance meets the DF2-5 bar (0.6205 weighted at np1 on champion 5c278648a;
#      gate at >= 0.58 to allow prompt-mix noise).
#   4. The speculation BOOST is present: DFlash2 decode t/s >= 1.5x the --spec-type none
#      control on the same build (DF2-5 measured 2.6x: 70.0 vs 26.6).
#
# Server flags replay the DF2-5 recipe verbatim (artifacts-df25/.../server_command.txt),
# only the binary path, port and instrumentation differ. GPU host threads on the SMT
# siblings 184-191 per canonical placement. Requires the GPU free or a held claim —
# this is a capability/ratio smoke, but the none-control ratio still wants a quiet device.
#
# Exit 0 = capability intact. Any other exit = a named refusal on stderr.

BUILD="${1:?usage: dflash2_capability_smoke.sh <build-dir> <out-dir>}"
OUT="${2:?usage: dflash2_capability_smoke.sh <build-dir> <out-dir>}"
TARGET=/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf
DRAFTER=/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
PORT=18211
ACCEPT_FLOOR=0.58
BOOST_FLOOR=1.5
N_PREDICT=384

mkdir -p "$OUT"
[[ -x "$BUILD/bin/llama-server" ]] || { echo "REFUSE: no llama-server in $BUILD/bin" >&2; exit 2; }
[[ -f "$TARGET" && -f "$DRAFTER" ]] || { echo "REFUSE: model or drafter missing" >&2; exit 2; }
# NOTE: capture first, match second. `... --help | grep -q` is a FALSE-REFUSAL trap under
# `set -o pipefail`: grep -q exits on the first match and closes the pipe, llama-server takes
# SIGPIPE (141), and pipefail reports the *successful* match as a failed pipeline.
HELP_TXT="$("$BUILD/bin/llama-server" --help 2>&1 || true)"
case "$HELP_TXT" in
    *draft-dflash*) ;;
    *) echo "REFUSE: build does not list draft-dflash in --spec-type" >&2; exit 2 ;;
esac

export LD_LIBRARY_PATH="$BUILD/bin"

SRV_PID=""
cleanup() {
    if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$SRV_PID" 2>/dev/null && kill -9 "$SRV_PID" 2>/dev/null || true
        wait "$SRV_PID" 2>/dev/null || true
    fi
    SRV_PID=""
}
trap cleanup EXIT

start_server() {  # start_server <arm> <extra flags...>
    local arm="$1"; shift
    taskset -c 184-191 "$BUILD/bin/llama-server" \
        -m "$TARGET" -np 1 -c 4096 -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 \
        --device ROCm0 -ngl 99 -fa on --host 127.0.0.1 --port $PORT --metrics --slots \
        "$@" > "$OUT/server-$arm.stdout" 2> "$OUT/server-$arm.stderr" &
    SRV_PID=$!
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && return 0
        kill -0 "$SRV_PID" 2>/dev/null || { echo "REFUSE($arm): server died during load — see $OUT/server-$arm.stderr" >&2; return 3; }
        sleep 2
    done
    echo "REFUSE($arm): server not healthy after 240s" >&2; return 3
}

run_completion() {  # run_completion <arm> -> writes response json, echoes "tokens t_per_s"
    local arm="$1"
    curl -sf "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' -d '{
        "prompt": "Prove that for every integer n >= 1, the sum 1^3 + 2^3 + ... + n^3 equals (n(n+1)/2)^2. Give the full induction argument, then compute the value for n = 12 and verify it directly.",
        "n_predict": '"$N_PREDICT"', "temperature": 0, "seed": 42, "cache_prompt": false
    }' > "$OUT/resp-$arm.json" || { echo "REFUSE($arm): completion request failed" >&2; return 4; }
    python3 - "$OUT/resp-$arm.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("timings", {})
n = t.get("predicted_n", 0)
tps = t.get("predicted_per_second", 0.0)
if n < 100:
    print(f"REFUSE: only {n} tokens generated", file=sys.stderr); sys.exit(4)
print(f"{n} {tps:.2f}")
EOF
}

# --- arm 1: DFlash2 -------------------------------------------------------------
start_server dflash -md "$DRAFTER" -ngld 99 --spec-type draft-dflash --spec-draft-n-max 8 --no-kv-unified
read -r DF_N DF_TPS <<< "$(run_completion dflash)"
ACCEPT=$(python3 - "$OUT/resp-dflash.json" "$OUT/server-dflash.stderr" <<'EOF'
import json, re, sys
d = json.load(open(sys.argv[1]))
t = d.get("timings", {})
da, dn = t.get("draft_n_accepted"), t.get("draft_n")
if isinstance(dn, (int, float)) and dn and dn > 0:
    print(f"{da/dn:.4f}"); sys.exit(0)
txt = open(sys.argv[2], errors="replace").read()
m = re.findall(r"draft acceptance rate\s*=\s*([0-9.]+)", txt)
if m:
    print(m[-1]); sys.exit(0)
print("REFUSE: no draft acceptance stats in timings or server stderr — was speculation active at all?", file=sys.stderr)
sys.exit(5)
EOF
)
cleanup

# --- arm 2: none control --------------------------------------------------------
start_server none --spec-type none
read -r NONE_N NONE_TPS <<< "$(run_completion none)"
cleanup

# --- verdict --------------------------------------------------------------------
python3 - "$OUT" "$ACCEPT" "$DF_TPS" "$NONE_TPS" "$DF_N" "$NONE_N" "$ACCEPT_FLOOR" "$BOOST_FLOOR" "$BUILD" <<'EOF'
import json, sys
out, accept, df_tps, none_tps, df_n, none_n, accept_floor, boost_floor, build = sys.argv[1:10]
accept, df_tps, none_tps = float(accept), float(df_tps), float(none_tps)
boost = df_tps / none_tps if none_tps > 0 else 0.0
verdict = {
    "schema": "epyc.dflash2_capability_smoke.v1",
    "build": build,
    "dflash": {"tokens": int(df_n), "tok_s": df_tps, "acceptance": accept},
    "none":   {"tokens": int(none_n), "tok_s": none_tps},
    "boost_x": round(boost, 3),
    "bars": {"acceptance_floor": float(accept_floor), "boost_floor": float(boost_floor),
             "reference": "DF2-5 np1 on 5c278648a: 70.0 t/s, acceptance 0.6205, boost 2.6x"},
    "passed": accept >= float(accept_floor) and boost >= float(boost_floor),
}
json.dump(verdict, open(f"{out}/verdict.json", "w"), indent=2)
print(json.dumps(verdict, indent=2))
if not verdict["passed"]:
    print(f"REFUSE: acceptance {accept:.4f} (floor {accept_floor}) boost {boost:.2f}x (floor {boost_floor}x)", file=sys.stderr)
    sys.exit(6)
EOF
echo "DFLASH2 CAPABILITY: PASS (acceptance $ACCEPT, ${DF_TPS} t/s vs ${NONE_TPS} t/s none)"
