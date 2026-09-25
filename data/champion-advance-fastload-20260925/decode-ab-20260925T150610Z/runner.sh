#!/bin/bash
# Champion-advance decode no-regression gate: ak/champion/llama-cpp-ffc1bac82eec 2b57340bf -> 90c12df42
#
#   A = baseline  /mnt/raid0/llm/kernels/builds/cpu-20260925-2b57340bf  (10306, current champion tip)
#   B = candidate /mnt/raid0/llm/kernels/builds/cpu-20260925-90c12df42  (10308, fastload)
#
# WHAT THIS GATE IS, AND WHERE EACH RULE COMES FROM (no codified CPU champion-advance gate exists:
# autokernel-rebuild-program.md:2833-2845 records the design as DEFERRED BY THE OPERATOR; this
# script assembles the nearest ratified rules):
#   scope      DERIVED, not quoted: .claude/skills/kernel-promotion/scripts/scope.sh --backend cpu
#              (2026-09-25: 8 CPU roles over 2 GGUFs, both draft-mtp) -> the two models below.
#              Snapshot: /mnt/raid0/llm/tmp/fastload-20260925/scope-cpu-20260925.json
#   recipe     the PRODUCTION serving recipe, spec decode ON (headline/production rule,
#              MEASUREMENT.md:164-170, bench-cpu.md P-BENCH-PLACEMENT-1 gate 6): argv and env
#              copied from the LIVE production servers :8070 (frontdoor) and :8074
#              (architect_critic) on 2026-09-25, which match stack_priors.yaml. Deltas, all
#              non-kernel: --port, no --slot-save-path (would write the production KV cache dir),
#              binary dir. Envelope = production SERVE_PREFIX `taskset -c 0-95 numactl
#              --interleave=all` (qwen38_flash_next_recipe.py:510). NOT llama-bench.
#   instrument server-native decode, P-BENCH-4 request shape (bench-cpu.md:184-189):
#              stream:false, cache_prompt:false, max_tokens 512, ignore_eos, temperature 0;
#              64-token warmup until 3 consecutive rates within 5% of their median (<=8 tries);
#              then exactly 5 measured requests; per-launch value = median predicted_per_second.
#   unit       process: a FRESH server per sample (promotion_gates.yaml speed.unit, serving_gate.py).
#   pairs      PAIRS (default 5 = promotion_gates.yaml cpu.n_per_arm; >=3 rounds per the host-drift
#              rule), strictly alternating A B A B ... inside ONE region-lock hold.
#   threshold  CPU kernel-promotion decision rule (bench-cpu.md:97-99; promotion_gates.yaml:50):
#              ratio = median(B launches) / median(A launches), higher-better tok/s, unit process.
#              >= 0.98 PASS; < 0.95 FAIL; [0.95,0.98) -> one fresh REVERSED pair (B then A),
#              pool every launch, pooled ratio >= 0.98 or FAIL. The GPU serving floor
#              7.249% [6.024, 9.371] n=24 is NOT used: it is a GPU/27B/DFlash2 floor and using it
#              here would violate FLOOR-UNIT-1 (MEASUREMENT.md:522).
#   correctness each measured response's text is hashed; A and B must produce identical text per
#              request index (temp 0; bit-exactness already shown). A mismatch -> INVALID.
#   load time  measured SEPARATELY (launch -> /health 200, plus the loader's own lines) and labelled
#              "load_s (not gated)". Default LLAMA_ARG_LOAD_THREADS (unset = auto) on both arms,
#              exactly as production launches.
#   evidence   DURABLE root only (runbook P4; check_evidence_durability.py refuses
#              /mnt/raid0/llm/tmp): /mnt/raid0/llm/epyc-inference-research/data/
#              champion-advance-fastload-20260925/decode-ab-<ts>/. This runner copies itself there
#              with its sha256.
#
# SAFETY
#   * DRY-RUN BY DEFAULT: identity checks + preflight + plan + exact argv; launches nothing.
#   * --execute re-execs itself under `region-lock run --cpu-list 0-95 --role bench --timeout-s 0`
#     (the same flocks orchestrator dispatch takes, so production CPU inference queues behind the
#     gate instead of contending with it; the whole ABAB is ONE hold).
#   * refuses to start if: any cpu_region lock is held by a PID outside this script's ancestry
#     (/proc/locks vs /mnt/raid0/llm/tmp/cpu_region.*.lock); any llama.cpp inference/measurement
#     process runs that is not a production-store server; the autokernel loop is alive; a production
#     CPU server is busy (CPU sampled over 5 s); MemAvailable is too low.
#   * samples foreign CPU DURING every measured window (/proc/stat minus our server); a run with
#     foreign load above FOREIGN_MAX_PCT invalidates the gate (verdict INVALID, not PASS/FAIL).
#   * kills ONLY the server PIDs it launched (TERM -> KILL, verified dead with ps -p).
#
# usage: champion_decode_ab.sh [--execute] [--pairs N] [--models frontdoor,architect_critic]
set -euo pipefail

SELF=$(readlink -f "$0")
T=/mnt/raid0/llm/tmp/fastload-20260925
A_BUILD=/mnt/raid0/llm/kernels/builds/cpu-20260925-2b57340bf
B_BUILD=/mnt/raid0/llm/kernels/builds/cpu-20260925-90c12df42
A_VER="10306 (2b57340bf)"
B_VER="10308 (90c12df42)"
LINK=/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh
REGION_LOCK=/workspace/repos/epyc-orchestrator/scripts/region-lock
EVROOT=/mnt/raid0/llm/epyc-inference-research/data/champion-advance-fastload-20260925
PORT_BASE=18771
FOREIGN_MAX_PCT=${FOREIGN_MAX_PCT:-400}   # % of one CPU (400 = 4 busy cores) inside a measured window
PROD_BUSY_MAX_PCT=${PROD_BUSY_MAX_PCT:-50}
HOST_BUSY_MAX_PCT=${HOST_BUSY_MAX_PCT:-400}   # whole host, % of one CPU, sampled 5 s before start
MEM_MARGIN_GB=64

EXECUTE=0; PAIRS=5; MODELS=frontdoor,architect_critic
while [ $# -gt 0 ]; do
    case "$1" in
        --execute) EXECUTE=1; shift ;;
        --pairs) PAIRS=$2; shift 2 ;;
        --models) MODELS=$2; shift 2 ;;
        *) echo "usage: $0 [--execute] [--pairs N] [--models frontdoor,architect_critic]"; exit 64 ;;
    esac
done
[ "$PAIRS" -ge 3 ] || { echo "REFUSE: --pairs must be >= 3 (host-drift rule)"; exit 64; }

# ---------------------------------------------------------------- production recipe (live argv)
PROD_ENV=(OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
          KMP_BLOCKTIME=10 GGML_IQK=1)
PREFIX=(taskset -c 0-95 numactl --interleave=all)
model_args() {   # $1 = model key -> prints production argv (without binary/port)
    case "$1" in
        frontdoor)  # live :8070 (Qwen3.6-35B-A3B-MTP-Q8_0, full instance)
            echo "-m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -np 4 -c 262144 -t 96 -ub 2048"\
                 "--flash-attn on --jinja -ctk q8_0 -ctv q8_0 --mlock --no-mmap --cache-ram 32768"\
                 "--chat-template-file /mnt/raid0/llm/models/chat-templates/epyc-qwen3x-v1-terse.jinja"\
                 "--spec-type draft-mtp --spec-draft-n-max 4 --reasoning off --device none --device-draft none" ;;
        architect_critic)  # live :8074 (Qwen3.8-Flash-Next UD-IQ4_XS + shared MTP drafter)
            echo "-m /mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"\
                 "-np 1 -c 262144 -t 96 -ub 2048 --flash-attn on --jinja -ctk f16 -ctv f16 --mlock --no-mmap"\
                 "--cache-ram 32768 -md /mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"\
                 "--spec-type draft-mtp --spec-draft-n-max 4 --draft-p-min 0.5 --reasoning off --device none --device-draft none" ;;
        *) echo "unknown model $1" >&2; return 1 ;;
    esac
}
model_mem_gb() { case "$1" in frontdoor) echo 60 ;; architect_critic) echo 110 ;; esac; }
PROMPT="Write a detailed technical explanation of how a CPU memory hierarchy works, covering registers, L1, L2 and L3 caches, main memory and NUMA, with concrete latency figures and the reasons each level exists."

# ---------------------------------------------------------------- helpers
say() { echo "[$(date -u +%H:%M:%S)] $*"; }
REFUSALS=()
refuse() { REFUSALS+=("$*"); say "REFUSE: $*"; }

ancestors() {   # PPid from /proc/<pid>/status (stat's comm field may contain spaces)
    local p=$$; while [ -n "$p" ] && [ "$p" -gt 1 ]; do echo "$p"; p=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null); done; }

check_identity() {
    local b=$1 want=$2 got
    got=$(env -u LD_LIBRARY_PATH "$b/bin/llama-server" --version 2>&1 | awk '/^version:/{print $2, $3}')
    [ "$got" = "$want" ] && say "OK   $b version $got" || refuse "$b --version '$got' != '$want'"
    ( cd "$b" && sha256sum -c --quiet SHA256SUMS ) >/dev/null 2>&1 && say "OK   $b SHA256SUMS" || refuse "$b SHA256SUMS mismatch"
    local rp; rp=$(readelf -d "$b/bin/llama-server" | awk -F'[][]' '/RUNPATH/{print $2}')
    [ "$rp" = '$ORIGIN' ] && say "OK   $b RUNPATH [\$ORIGIN]" || refuse "$b RUNPATH [$rp]"
    env -u LD_LIBRARY_PATH "$LINK" "$b/bin/llama-server" "$b/bin" >/dev/null 2>&1 \
        && say "OK   $b verify_ggml_linkage PASS" || refuse "$b verify_ggml_linkage FAIL"
}

check_locks() {   # any cpu_region lock held by a PID outside our ancestry
    local anc; anc=" $(ancestors | tr '\n' ' ') "
    local f key holders
    for f in /mnt/raid0/llm/tmp/cpu_region.*.lock; do
        [ -e "$f" ] || continue
        local d i; d=$(stat -c '%d' "$f"); i=$(stat -c '%i' "$f")
        key=$(printf '%02x:%02x:%s' $(( (d>>8)&0xfff )) $(( (d&0xff) | ((d>>12)&0xfff00) )) "$i")
        holders=$(awk -v k="$key" '$6==k{print $5}' /proc/locks | sort -u)
        for h in $holders; do
            case "$anc" in *" $h "*) continue ;; esac
            refuse "lock $(basename "$f") held by pid $h: $(ps -o args= -p "$h" 2>/dev/null | cut -c1-140)"
        done
    done
}

prod_dirs() { readlink -f /mnt/raid0/llm/kernels/production/cpu; readlink -f /mnt/raid0/llm/kernels/production/gpu; }
check_processes() {
    local pd; pd=$(prod_dirs | tr '\n' ' ')
    local p exe name
    for p in /proc/[0-9]*; do
        name=$(cat "$p/comm" 2>/dev/null) || continue
        case "$name" in
            llama-*|test-backend-op*|test-backend-ops) ;;
            *) continue ;;
        esac
        exe=$(readlink -f "$p/exe" 2>/dev/null || echo "?")
        case "$name" in
            llama-server)
                local ok=0 d
                for d in $pd; do [ "$(dirname "$exe")" = "$d" ] && ok=1; done
                [ $ok -eq 1 ] && continue
                refuse "non-production llama-server pid ${p#/proc/} exe=$exe" ;;
            *) refuse "inference/measurement process pid ${p#/proc/} $name exe=$exe" ;;
        esac
    done
    if pgrep -f 'scripts\.kernel_rnd\.autokernel\.loop\.run' >/dev/null 2>&1; then
        refuse "autokernel loop is running: $(pgrep -af 'scripts\.kernel_rnd\.autokernel\.loop\.run' | cut -c1-160 | head -2)"
    fi
}

prod_cpu_pids() {   # production CPU llama-servers
    local cdir; cdir=$(readlink -f /mnt/raid0/llm/kernels/production/cpu)
    local p; for p in /proc/[0-9]*; do
        [ "$(cat "$p/comm" 2>/dev/null)" = llama-server ] || continue
        [ "$(dirname "$(readlink -f "$p/exe" 2>/dev/null)")" = "$cdir" ] && echo "${p#/proc/}"
    done
}
ticks() { local s=0 p; for p in "$@"; do [ -r "/proc/$p/stat" ] && s=$((s + $(awk '{print $14+$15}' "/proc/$p/stat"))); done; echo $s; }
check_prod_idle() {
    local pids; pids=$(prod_cpu_pids | tr '\n' ' ')
    local hz; hz=$(getconf CLK_TCK)
    local t0 t1; t0=$(ticks $pids); sleep 5; t1=$(ticks $pids)
    local pct=$(( (t1 - t0) * 100 / hz / 5 ))
    say "production CPU servers [$pids] used ${pct}% CPU over 5 s (max ${PROD_BUSY_MAX_PCT}%)"
    [ "$pct" -le "$PROD_BUSY_MAX_PCT" ] || refuse "production CPU servers busy: ${pct}% over 5 s"
}

check_host_quiet() {   # whole-host busy CPU over 5 s, any process (a compile, a node job, ...)
    local b0 a0 b1 a1
    read -r b0 a0 < <(awk '/^cpu /{s=0; for(i=2;i<=NF;i++) s+=$i; print s-$5-$6, s}' /proc/stat)
    local p0; p0=$(for p in /proc/[0-9]*; do awk -v p="${p#/proc/}" '{sub(/.*\) /,""); split($0,f," "); print p, f[12]+f[13]}' "$p/stat" 2>/dev/null; done)
    sleep 5
    read -r b1 a1 < <(awk '/^cpu /{s=0; for(i=2;i<=NF;i++) s+=$i; print s-$5-$6, s}' /proc/stat)
    local pct=$(( (b1 - b0) * 192 * 100 / (a1 - a0) ))
    say "host busy CPU over 5 s: ${pct}% of one CPU (max ${HOST_BUSY_MAX_PCT}%)"
    if [ "$pct" -gt "$HOST_BUSY_MAX_PCT" ]; then
        refuse "host not quiet: ${pct}% busy over 5 s; top consumers in the window:"
        local hz; hz=$(getconf CLK_TCK)
        while read -r pid t; do
            [ -r "/proc/$pid/stat" ] || continue
            local t1; t1=$(awk '{sub(/.*\) /,""); split($0,f," "); print f[12]+f[13]}' "/proc/$pid/stat" 2>/dev/null) || continue
            echo "$(( (t1 - t) * 100 / hz / 5 )) $pid"
        done <<< "$p0" | sort -rn | head -5 | while read -r c pid; do
            say "      ${c}%  pid $pid  $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-120)"; done
    fi
}

check_mem() {
    local need=0 m; for m in ${MODELS//,/ }; do [ "$(model_mem_gb "$m")" -gt "$need" ] && need=$(model_mem_gb "$m"); done
    need=$((need + MEM_MARGIN_GB))
    local avail; avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    say "MemAvailable ${avail} GB (need >= ${need} GB for one server + margin)"
    [ "$avail" -ge "$need" ] || refuse "MemAvailable ${avail} GB < ${need} GB"
}

host_state() {
    echo "uptime: $(uptime)"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    echo "thp: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null)"
    echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing 2>/dev/null)"
    grep -E 'MemTotal|MemAvailable|AnonHugePages' /proc/meminfo
}

preflight() {
    REFUSALS=()
    say "--- identity"; check_identity "$A_BUILD" "$A_VER"; check_identity "$B_BUILD" "$B_VER"
    say "--- cpu_region locks"; check_locks
    say "--- competing processes"; check_processes
    say "--- production CPU servers idle"; check_prod_idle
    say "--- host quiet"; check_host_quiet
    say "--- memory"; check_mem
    say "--- region-lock status"; "$REGION_LOCK" status 2>&1 | sed 's/^/    /' || true
    [ ${#REFUSALS[@]} -eq 0 ]
}

server_cmd() {   # $1 build $2 model $3 port -> full argv on stdout (one line)
    echo "env -i HOME=$HOME PATH=/usr/bin:/bin ${PROD_ENV[*]} LD_LIBRARY_PATH=$1/bin:/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/opt/rocm/lib" \
         "${PREFIX[*]} $1/bin/llama-server $(model_args "$2") --host 127.0.0.1 --port $3"
}

# ---------------------------------------------------------------- one launch = one sample
SERVER_PID=""
stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 120); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$SERVER_PID" 2>/dev/null; then kill -KILL "$SERVER_PID" 2>/dev/null || true; sleep 2; fi
    if ps -p "$SERVER_PID" >/dev/null 2>&1; then say "ERROR: own server pid $SERVER_PID still alive"; exit 3; fi
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap 'stop_server' EXIT
trap 'stop_server; exit 130' INT TERM

run_launch() {   # $1 arm(A|B) $2 build $3 model $4 seq -> appends one JSON line to $OUT/launches.jsonl
    local arm=$1 b=$2 m=$3 seq=$4
    local port=$((PORT_BASE + seq % 2))
    local tag="${m}-$(printf %02d "$seq")-$arm"
    local log=$OUT/$tag.server.log
    local cmd; cmd=$(server_cmd "$b" "$m" "$port")
    echo "$cmd" > "$OUT/$tag.argv"
    local hz; hz=$(getconf CLK_TCK)
    local t0; t0=$(date +%s.%N)
    bash -c "exec $cmd" > "$log" 2>&1 &
    SERVER_PID=$!
    local ok=0
    for _ in $(seq 1 1200); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        curl -sf -o /dev/null "http://127.0.0.1:$port/health" && { ok=1; break; }
        sleep 0.25
    done
    local t1; t1=$(date +%s.%N)
    [ $ok -eq 1 ] || { say "server $tag never became healthy"; tail -20 "$log"; stop_server; return 1; }
    # residency: ggml mapped from THIS build only
    if grep -E 'libggml|libllama' "/proc/$SERVER_PID/maps" | awk '{print $6}' | sort -u | grep -v "^$b/bin/" | grep -q .; then
        say "ggml mapped from outside $b/bin"; stop_server; return 1
    fi
    local rc=0
    python3 - "$port" "$OUT/$tag.requests.json" "$SERVER_PID" "$hz" "$PROMPT" <<'PY' || rc=$?
import json, statistics, sys, time, urllib.request
port, out, spid, hz, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
def cpu_total():
    with open("/proc/stat") as f: v = [int(x) for x in f.readline().split()[1:]]
    return sum(v) - v[3] - v[4], sum(v)          # busy, all
def own():
    import os
    t = 0
    for tid in os.listdir(f"/proc/{spid}/task"):
        try:
            with open(f"/proc/{spid}/task/{tid}/stat") as f: s = f.read().rsplit(")", 1)[1].split()
            t += int(s[11]) + int(s[12])
        except OSError: pass
    return t
def req(n):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n, "temperature": 0,
            "top_k": 1, "seed": 42, "stream": False, "cache_prompt": False, "ignore_eos": True}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    b0, a0 = cpu_total(); o0 = own(); w0 = time.time()
    with urllib.request.urlopen(r, timeout=1800) as resp: d = json.load(resp)
    w1 = time.time(); b1, a1 = cpu_total(); o1 = own()
    ncpu = 192
    foreign_pct = max(0.0, ((b1 - b0) - (o1 - o0)) / max(1, a1 - a0) * ncpu * 100)
    t = d.get("timings", {})
    c = d["choices"][0]
    return {"n": n, "wall_s": w1 - w0, "finish_reason": c.get("finish_reason"),
            "predicted_n": t.get("predicted_n"), "predicted_per_second": t.get("predicted_per_second"),
            "prompt_n": t.get("prompt_n"), "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
            "foreign_cpu_pct": round(foreign_pct, 1),
            "text_sha256": __import__("hashlib").sha256(c["message"]["content"].encode()).hexdigest(), "raw": d}
warm = []
for _ in range(8):
    w = req(64); warm.append(w)
    rates = [x["predicted_per_second"] for x in warm[-3:] if x["predicted_per_second"]]
    if len(rates) == 3:
        med = statistics.median(rates)
        if all(abs(r - med) / med <= 0.05 for r in rates): break
else:
    json.dump({"error": "warmup never stabilised", "warmup": warm}, open(out, "w")); sys.exit(2)
meas = [req(512) for _ in range(5)]
bad = [m for m in meas if m["finish_reason"] != "length" or m["predicted_n"] != 512 or not (m["predicted_per_second"] or 0) > 0]
json.dump({"warmup": warm, "measured": meas, "invalid": len(bad)}, open(out, "w"), indent=1)
sys.exit(3 if bad else 0)
PY
    local loadlines; loadlines=$(grep -E 'load_all_data_parallel|load time|model loaded' "$log" | tr '\n' ' ' | cut -c1-400 || true)
    stop_server
    python3 - "$OUT/$tag.requests.json" "$arm" "$m" "$seq" "$b" "$(echo "$t1 - $t0" | bc)" "$rc" "$loadlines" >> "$OUT/launches.jsonl" <<'PY'
import json, statistics, sys
f, arm, m, seq, b, load_s, rc, loadlines = sys.argv[1:9]
d = json.load(open(f))
meas = d.get("measured", [])
rates = [x["predicted_per_second"] for x in meas if x.get("predicted_per_second")]
print(json.dumps({"arm": arm, "model": m, "seq": int(seq), "build": b, "rc": int(rc),
    "decode_tok_s_median": statistics.median(rates) if rates else None, "decode_tok_s": rates,
    "foreign_cpu_pct_max": max([x["foreign_cpu_pct"] for x in meas] or [None]),
    "text_sha256": [x["text_sha256"] for x in meas],
    "load_s_not_gated": float(load_s), "loader_lines": loadlines}))
PY
    say "$tag rc=$rc load_s=$(echo "$t1 - $t0" | bc) $(tail -1 "$OUT/launches.jsonl" | python3 -c 'import json,sys;d=json.load(sys.stdin);print("decode_median=%s foreign_max=%s%%"%(d["decode_tok_s_median"],d["foreign_cpu_pct_max"]))')"
    return 0
}

summarize() {   # prints the verdict; exit code 0 PASS, 1 FAIL, 2 INVALID, 4 GRAY (needs reversed pair)
    python3 - "$OUT/launches.jsonl" "$FOREIGN_MAX_PCT" "$1" "$PAIRS" <<'PY'
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
fmax = float(sys.argv[2]); model = sys.argv[3]; pairs = int(sys.argv[4])
rows = [r for r in rows if r["model"] == model]
def mad(x):
    m = statistics.median(x); return statistics.median([abs(v - m) for v in x])
invalid = [r for r in rows if r["rc"] != 0 or r["decode_tok_s_median"] is None
           or (r["foreign_cpu_pct_max"] or 0) > fmax]
A = [r for r in rows if r["arm"] == "A"]; B = [r for r in rows if r["arm"] == "B"]
shaA = {tuple(r["text_sha256"]) for r in A}; shaB = {tuple(r["text_sha256"]) for r in B}
text_ok = len(shaA | shaB) == 1
a = [r["decode_tok_s_median"] for r in A if r["decode_tok_s_median"]]
b = [r["decode_tok_s_median"] for r in B if r["decode_tok_s_median"]]
la = [r["load_s_not_gated"] for r in A]; lb = [r["load_s_not_gated"] for r in B]
print(f"== {model}: n_launch A={len(A)} B={len(B)}")
for r in rows:
    print(f"   seq {r['seq']:>2} {r['arm']} decode_median={r['decode_tok_s_median']} tok/s  "
          f"foreign_max={r['foreign_cpu_pct_max']}%  load_s(not gated)={r['load_s_not_gated']:.1f}  rc={r['rc']}")
if a and b:
    ratio = statistics.median(b) / statistics.median(a)
    pa = [v for r in A for v in r["decode_tok_s"]]; pb = [v for r in B for v in r["decode_tok_s"]]
    print(f"   A decode median {statistics.median(a):.3f} tok/s MAD {mad(a):.3f} | "
          f"B {statistics.median(b):.3f} MAD {mad(b):.3f} | ratio B/A {ratio:.4f} (unit process)")
    print(f"   per-request pooled: A {statistics.median(pa):.3f} B {statistics.median(pb):.3f} "
          f"ratio {statistics.median(pb)/statistics.median(pa):.4f} (n={len(pa)}/{len(pb)}, diagnostic)")
    print(f"   load_s (NOT gated): A median {statistics.median(la):.1f} s | B median {statistics.median(lb):.1f} s | "
          f"A/B {statistics.median(la)/statistics.median(lb):.2f}x")
print(f"   text identical across all launches/arms: {text_ok}")
missing = max(0, pairs - len(A)) + max(0, pairs - len(B))   # a launch that never produced a row
if missing: print(f"   {missing} launch(es) produced no result row (never healthy / crashed)")
if invalid or missing or not text_ok or not a or not b:
    print(f"   VERDICT {model}: INVALID ({len(invalid)} invalid launches, text_ok={text_ok})"); sys.exit(2)
if ratio >= 0.98: print(f"   VERDICT {model}: PASS (ratio {ratio:.4f} >= 0.98)"); sys.exit(0)
if ratio < 0.95: print(f"   VERDICT {model}: FAIL (ratio {ratio:.4f} < 0.95)"); sys.exit(1)
print(f"   VERDICT {model}: GRAY (0.95 <= {ratio:.4f} < 0.98) -> reversed pair + pool"); sys.exit(4)
PY
}

est() {   # rough duration: per launch ~ load + warmup + 5x512 decode + teardown
    local total=0 m per
    for m in ${MODELS//,/ }; do
        case "$m" in frontdoor) per=150 ;; architect_critic) per=180 ;; esac
        total=$(( total + per * 2 * PAIRS ))
    done
    echo "$total"
}

# ================================================================ main
if [ "$EXECUTE" -eq 0 ]; then
    say "DRY RUN (nothing is launched). Plan: models=$MODELS pairs=$PAIRS order=A B A B ... (A=2b57340bf baseline, B=90c12df42 candidate)"
    host_state | sed 's/^/    /'
    if preflight; then say "PREFLIGHT: would START"; else say "PREFLIGHT: would REFUSE (${#REFUSALS[@]} reason(s))"; fi
    say "--- exact server argv per arm (fresh process per sample; ports $PORT_BASE/$((PORT_BASE+1)))"
    for m in ${MODELS//,/ }; do
        echo "  [$m] A: $(server_cmd "$A_BUILD" "$m" "$PORT_BASE")"
        echo "  [$m] B: $(server_cmd "$B_BUILD" "$m" "$((PORT_BASE+1))")"
    done
    say "--- request: POST /v1/chat/completions {max_tokens:512, temperature:0, top_k:1, seed:42, stream:false, cache_prompt:false, ignore_eos:true}; warmup 64 tok x(3..8); 5 measured"
    say "--- execute wrapper: $REGION_LOCK run --cpu-list 0-95 --role bench --timeout-s 0 --tag champion-decode-ab:<ts> -- env CHAMPION_AB_LOCKED=1 $SELF --execute --pairs $PAIRS --models $MODELS"
    s=$(est); say "--- expected duration ~$((s/60)) min (+ ~$(( s / PAIRS / 2 / 60 * 2 )) min per model if a GRAY reversed pair is needed)"
    say "--- evidence would go to $EVROOT/decode-ab-<ts>/"
    exit 0
fi

if [ -z "${CHAMPION_AB_LOCKED:-}" ]; then
    say "acquiring region-lock (all of 0-95, role bench) for the whole A/B"
    exec "$REGION_LOCK" run --cpu-list 0-95 --role bench --timeout-s 0 \
        --tag "champion-decode-ab:$(date -u +%Y%m%dT%H%M%SZ)" -- \
        env CHAMPION_AB_LOCKED=1 "$SELF" --execute --pairs "$PAIRS" --models "$MODELS"
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$EVROOT/decode-ab-$TS
mkdir -p "$OUT"
cp "$SELF" "$OUT/runner.sh"; sha256sum "$SELF" > "$OUT/runner.sha256"
exec > >(tee -a "$OUT/run.log") 2>&1
say "evidence: $OUT"
host_state > "$OUT/host_state.before.txt"
if ! preflight; then say "PREFLIGHT REFUSED -- nothing launched"; printf '%s\n' "${REFUSALS[@]}" > "$OUT/REFUSED"; exit 2; fi
: > "$OUT/launches.jsonl"

overall=0; seq=0
for m in ${MODELS//,/ }; do
    for i in $(seq 1 "$PAIRS"); do
        seq=$((seq+1)); run_launch A "$A_BUILD" "$m" "$seq" || true
        seq=$((seq+1)); run_launch B "$B_BUILD" "$m" "$seq" || true
    done
    set +e; summarize "$m" | tee "$OUT/summary.$m.txt"; rc=${PIPESTATUS[0]}; set -e
    if [ "$rc" -eq 4 ]; then
        say "$m GRAY -> one fresh reversed pair (B then A), pooled"
        seq=$((seq+1)); run_launch B "$B_BUILD" "$m" "$seq" || true
        seq=$((seq+1)); run_launch A "$A_BUILD" "$m" "$seq" || true
        set +e; summarize "$m" | tee "$OUT/summary.$m.txt"; rc=${PIPESTATUS[0]}; set -e
        [ "$rc" -eq 4 ] && rc=1   # pooled ratio still < 0.98 -> FAIL (bench-cpu.md:98-99)
    fi
    [ "$rc" -gt "$overall" ] && overall=$rc
done
check_locks; host_state > "$OUT/host_state.after.txt"
case "$overall" in 0) v=PASS ;; 1) v=FAIL ;; *) v=INVALID ;; esac
say "OVERALL VERDICT: $v"
echo "$v" > "$OUT/VERDICT"
( cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
exit $overall
