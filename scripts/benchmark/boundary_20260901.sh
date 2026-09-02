#!/bin/bash
# boundary_20260901.sh — unattended driver for the AutoKernel run-23→24 boundary.
#
#   boundary_20260901.sh                # wait until 2026-09-01 22:00Z, then run
#   boundary_20260901.sh --now          # skip the wait, run immediately
#   boundary_20260901.sh --dry-run      # print the resolved plan + validate every
#                                       # precondition checkable NOW; execute nothing
#
# Modeled on boundary_20260831.sh (the run-21→22 driver) — same state-file resume,
# per-step evidence files, fail-closed refusals, recorded-PID-only kill discipline.
# Sequence (each step HARD-GATED on the previous step's green — stricter than the
# 20260831 driver's log-and-step-past policy, per the boundary spec):
#
#   step0  WAIT           sleep-loop until T0 (2026-09-01 22:00Z), resumable
#   step1  STOP run 23    SIGTERM the recorded pid (cmdline-verified), 15 min grace,
#                         SIGKILL escalation, verify dead; record final tally
#   step2  MERGE GATE     rung commits + review token required; merge
#                         ak/rung-fixes-20260901 into lane/ak-rebuild-20260828 IN THE
#                         LANE (the one legal lane-edit moment: loop verified dead);
#                         hardware-free gate suite post-merge; any red = rollback+refuse
#   step3  SEEDS          staged re-anchored seeds -> store inbox, originals backed up
#   step4  ROCPROF        dispatch-table sanity per confirm surface (dec-b4, dec-b8)
#                         on the 27B production model, FROZEN production build
#   step5  DFLASH2 SMOKE  llama-bench on the DFlash2 drafter head; informs D5,
#                         does NOT gate (the only non-gating step)
#   step6  27B A/A CALIB  keyed floors calibration/<surface>.<model-stem>.json for
#                         dec-b4 + dec-b8 with the POST-MERGE lane code (long step)
#   step7  READINESS      package + all-green verdict; launches run 24 ONLY if the
#                         PREAUTH_RUN24 token exists AND every gating step is green
#                         (the run-22 watcher mechanism, folded in)
#
# Deviations from the 20260831 prior art, each deliberate:
#   * set -euo pipefail (spec) instead of set -uo: gating is strict here, so an
#     unguarded failure SHOULD kill the driver; state-file resume re-enters at the
#     incomplete step. Steps that tolerate failure (step5) guard their own rc.
#   * NO STOP sentinel is written into the store: the boundary spec forbids touching
#     loop-memory outside the scripted writes (inbox copies, calibration), and
#     SIGTERM alone is a documented loop stop path (run.py --iterations help).
#   * A missing run23.pid file is a REFUSAL, not an assumed-stopped: step 2 edits
#     the lane, which is only legal over a VERIFIED-dead loop.
#   * The calibration anchor is resolved at run time to the LATEST provenance-carrying
#     anchor-gen-* in the store (pinning anchor-gen-005 is what forced the 20260831
#     watcher's 21:40Z rewrite after the loop pruned it).
#
# Kill discipline: signals go only to the pid self-read from run23.pid (and its
# process group resolved FROM that pid); never a name-pattern kill. The /proc scan
# after a run-24 launch is read-only (the watcher's own pattern).
# Claim discipline: device steps hold mi210_0 via autokernel.loop.claim.hold inside
# the helper (the serving_evidence_refresh.py pattern); this driver NEVER touches
# /mnt/raid0/llm/tmp/gpu_device.mi210_0.lock itself.
#
# Test seams (all default to production values):
#   BOUNDARY_STUB_DIR      dir of executable stubs replacing heavy commands
#   BOUNDARY_WORK_DIR / BOUNDARY_PID_FILE / BOUNDARY_STORE / BOUNDARY_LANE_ROOT
#   BOUNDARY_T0            "YYYY-MM-DD HH:MM" UTC (default 2026-09-01 22:00)
#   BOUNDARY_KILL_WAIT_S / BOUNDARY_KILL_GRACE_S / BOUNDARY_POLL_S
#   BOUNDARY_CALIB_PAIRS / BOUNDARY_CALIB_ANCHOR
#   BOUNDARY_LIB_ONLY=1    source the file for its functions, run nothing
set -euo pipefail

# ---------------------------------------------------------------- configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/boundary_20260901_helper.py"

LANE_ROOT="${BOUNDARY_LANE_ROOT:-/mnt/raid0/llm/worktrees/mains/ak-rebuild-research}"
LANE_BRANCH="lane/ak-rebuild-20260828"
RUNG_BRANCH="ak/rung-fixes-20260901"
RUNG_MSG_PREFIX="autokernel rung:"
PRE_MERGE_TAG="ak/pre-rung-merge-20260901"

WORK_DIR="${BOUNDARY_WORK_DIR:-/mnt/raid0/llm/tmp/boundary-20260901}"
STATE_FILE="$WORK_DIR/state"
PID_FILE="${BOUNDARY_PID_FILE:-/mnt/raid0/llm/tmp/run23.pid}"
RUN23_LOG="${BOUNDARY_RUN23_LOG:-/mnt/raid0/llm/tmp/run23.log}"
STORE="${BOUNDARY_STORE:-/mnt/raid0/llm/autokernel/loop-memory}"
INBOX="$STORE/inbox"
SEED_STAGING="${BOUNDARY_SEED_STAGING:-/mnt/raid0/llm/tmp/seeds-reanchored-20260901}"
INBOX_BACKUP="$WORK_DIR/inbox-backup"
REVIEW_TOKEN="$WORK_DIR/REVIEW_TOKEN_R23_11"
PREAUTH_TOKEN="$WORK_DIR/PREAUTH_RUN24"
READINESS="$WORK_DIR/READINESS.md"
RUNG_IDENTITY_DIR="$WORK_DIR/rung-identity"

# Production frozen build = the reference dispatch table (v9 freeze; verified, never built).
PROD_TREE="${BOUNDARY_PROD_TREE:-/mnt/raid0/llm/llama.cpp}"
PROD_BRANCH="production-consolidated-v9"
PROD_COMMIT="0db32c06e3e550065b78311a6031ef3dd2c4f27c"
PROD_BENCH="$PROD_TREE/build-hip/bin/llama-bench"

# 27B production model: the constant serving_evidence_refresh.py measures through
# (champion_anchor_validation.py:48 MODEL — the refresh drives that harness).
MODEL_27B="${BOUNDARY_MODEL_27B:-/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf}"
MODEL_SCREEN="${BOUNDARY_MODEL_SCREEN:-/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
DFLASH2_GGUF="${BOUNDARY_DFLASH2_GGUF:-/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf}"

CHAMP_TREE="${BOUNDARY_CHAMP_TREE:-/mnt/raid0/llm/tmp/champ2}"
T0_UTC="${BOUNDARY_T0:-2026-09-01 22:00}"
KILL_WAIT_S="${BOUNDARY_KILL_WAIT_S:-900}"     # 15 min grace (drain-tier loop: ≤5 min historically)
KILL_GRACE_S="${BOUNDARY_KILL_GRACE_S:-60}"
POLL_S="${BOUNDARY_POLL_S:-30}"
LOOP_MATCH_A="autokernel.loop.run"
LOOP_MATCH_B="--surface dec-b4"
CALIB_PAIRS="${BOUNDARY_CALIB_PAIRS:-20}"      # same N the 20260831 boundary calibrated with
CONFIRM_SURFACES=(dec-b4 dec-b8)
MODEL_27B_STEM="$(basename "$MODEL_27B" .gguf)"

RUN24_OUT="$STORE/run24"
RUN24_LOG="/mnt/raid0/llm/tmp/run24.log"
RUN24_PIDFILE="/mnt/raid0/llm/tmp/run24.pid"

# ROCm toolchain env, mirrored from the 20260831 driver (read from run 21's live env).
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-/opt/rocm}"
case ":$PATH:" in *":/opt/rocm/bin:"*) ;; *) export PATH="/opt/rocm/bin:$PATH" ;; esac
export LD_LIBRARY_PATH="/opt/AMD/aocc-compiler-5.0.0/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DRY_RUN=0

# ---------------------------------------------------------------- plumbing
log() {
    local line
    line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    echo "$line"
    [[ "$DRY_RUN" == 1 ]] || echo "$line" >> "$WORK_DIR/driver.log"
}

refuse() {  # refuse <step> <exactly what was missing>
    local step="$1"; shift
    log "$step: REFUSED — $*"
    state_set "${step}_refused" "$*"
    exit 1
}

state_done() { [[ -f "$STATE_FILE" ]] && grep -q "^$1=done$" "$STATE_FILE"; }
state_mark() { echo "$1=done" >> "$STATE_FILE"; }
state_set()  { [[ "$DRY_RUN" == 1 ]] || echo "$1=$2" >> "$STATE_FILE"; }
state_get()  { { [[ -f "$STATE_FILE" ]] && grep "^$1=" "$STATE_FILE" | tail -1 | cut -d= -f2-; } || true; }

heavy() {  # heavy <stub-name> <logfile> <argv...>  — the stub seam for testing
    local name="$1" lf="$2"; shift 2
    if [[ -n "${BOUNDARY_STUB_DIR:-}" && -x "${BOUNDARY_STUB_DIR}/$name" ]]; then
        "${BOUNDARY_STUB_DIR}/$name" "$@" >> "$lf" 2>&1
    else
        "$@" >> "$lf" 2>&1
    fi
}

pid_alive() { [[ -d "/proc/$1" ]]; }

read_cmdline() {  # read_cmdline <pid> — argv joined by spaces, empty if unreadable
    tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null || true
}

loop23_cmdline_ok() {  # pid-recycling guard: cmdline must carry BOTH markers
    local cl; cl=$(read_cmdline "$1")
    grep -qF "$LOOP_MATCH_A" <<< "$cl" && grep -qF -- "$LOOP_MATCH_B" <<< "$cl"
}

lane_git() { git -C "$LANE_ROOT" "$@"; }

latest_anchor_gen() {  # newest anchor-gen-* in the store that carries provenance.json
    local d best=""
    for d in "$STORE"/anchor-gen-*/; do
        [[ -f "$d/provenance.json" ]] && best="$d"
    done
    [[ -n "$best" ]] && echo "${best%/}"
}

# ---------------------------------------------------------------- step 0: wait for T0
step0_wait() {
    if state_done step0; then log "step0: already done, skipping"; return 0; fi
    local t0 now
    t0=$(state_get t0_epoch)
    if [[ -z "$t0" ]]; then
        t0=$(date -u -d "$T0_UTC" +%s) || refuse step0 "cannot parse BOUNDARY_T0 '$T0_UTC'"
        state_set t0_epoch "$t0"   # pinned: a crash+relaunch past midnight keeps ONE T0
    fi
    now=$(date -u +%s)
    if (( now >= t0 )); then
        log "step0: T0 (${T0_UTC}Z) already reached — proceeding"
    else
        log "step0: waiting $(( t0 - now ))s until ${T0_UTC}Z (resumable; poll ${POLL_S}s)"
        while now=$(date -u +%s); (( now < t0 )); do
            sleep "$POLL_S"
        done
        log "step0: T0 reached"
    fi
    state_mark step0
}

# ---------------------------------------------------------------- step 1: stop run 23
step1_stop_run23() {
    if state_done step1; then log "step1: already done, skipping"; return 0; fi
    local lf="$WORK_DIR/step1-stop.log" pid pgid waited remnants tally="$WORK_DIR/step1-run23-final-tally.txt"
    log "step1: stopping run 23 (log: $lf)"

    [[ -f "$PID_FILE" ]] || refuse step1 "pid file $PID_FILE is missing — cannot VERIFY run 23 dead, and the step-2 lane merge is only legal over a verified-dead loop. If run 23 was stopped by hand, recreate the pid file with the old pid (a dead pid verifies clean) and relaunch."
    pid=$(tr -dc '0-9' < "$PID_FILE")
    [[ -n "$pid" ]] || refuse step1 "pid file $PID_FILE holds no numeric pid"
    state_set run23_pid "$pid"

    if ! pid_alive "$pid"; then
        log "step1: pid $pid already dead — run 23 already stopped"
        state_set step1_outcome "already_dead"
    else
        # Pid-recycling guard: refuse to signal a live pid that is not run 23.
        loop23_cmdline_ok "$pid" || refuse step1 "pid $pid is ALIVE but its cmdline '$(read_cmdline "$pid" | head -c 200)' does not contain BOTH '$LOOP_MATCH_A' and '$LOOP_MATCH_B' — the pid may have been recycled; refusing to signal it"
        kill -TERM "$pid" 2>> "$lf" && log "step1: SIGTERM sent to $pid" \
            || log "step1: WARNING SIGTERM to $pid failed (raced exit?)"

        waited=0   # graceful drain: loop stops at actor boundaries, historically ≤5 min
        while (( waited < KILL_WAIT_S )) && pid_alive "$pid"; do
            sleep "$POLL_S"; waited=$(( waited + POLL_S ))
        done

        if pid_alive "$pid"; then
            pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -dc '0-9') || true
            if [[ -n "${pgid:-}" && "$pgid" != "$(ps -o pgid= -p $$ | tr -dc '0-9')" ]]; then
                log "step1: still alive after ${KILL_WAIT_S}s — SIGKILL process group $pgid"
                kill -KILL -- "-$pgid" 2>> "$lf" || true
                state_set step1_escalated "sigkill_group_$pgid"
            else
                log "step1: WARNING pgid '${pgid:-}' unusable or equals ours — SIGKILL pid only"
                kill -KILL "$pid" 2>> "$lf" || true
                state_set step1_escalated "sigkill_pid"
            fi
            waited=0
            while (( waited < KILL_GRACE_S )) && pid_alive "$pid"; do
                sleep "$POLL_S"; waited=$(( waited + POLL_S ))
            done
        else
            state_set step1_escalated "none_sigterm_sufficed"
        fi

        pid_alive "$pid" && refuse step1 "pid $pid survived SIGKILL — run 23 is unkillable; no lane edit or device step may proceed"
        log "step1: pid $pid confirmed dead (/proc/$pid gone)"
        state_set step1_outcome "stopped"
    fi

    # Group remnants: observed by PGID of the self-read pid — never a name pattern.
    pgid=$(state_get step1_escalated | grep -o '[0-9]*$') || true
    pgid="${pgid:-$pid}"   # setsid'd loop: pgid == pid
    remnants=$(ps -eo pid=,pgid= | awk -v g="$pgid" '$2==g {print $1}') || true
    if [[ -n "$remnants" ]]; then
        log "step1: group $pgid remnants: $(echo "$remnants" | tr '\n' ' ')— SIGKILL group"
        kill -KILL -- "-$pgid" 2>> "$lf" || true
        sleep "$POLL_S"
        remnants=$(ps -eo pid=,pgid= | awk -v g="$pgid" '$2==g {print $1}') || true
        [[ -n "$remnants" ]] && refuse step1 "process-group $pgid remnants persist after SIGKILL: $remnants"
    fi
    state_set step1_group_remnants "none"

    # Final run-23 tally: last summary lines of its log + loop-status dispositions (read-only).
    {
        echo "=== run-23 final log tail ($RUN23_LOG) ==="
        [[ -f "$RUN23_LOG" ]] && tail -40 "$RUN23_LOG" || echo "(run-23 log missing)"
        echo "=== final loop-status dispositions ==="
        if [[ -f "$STORE/loop-status.json" ]]; then
            python3 -m json.tool "$STORE/loop-status.json" 2>/dev/null || cat "$STORE/loop-status.json"
        else
            echo "(no loop-status.json in $STORE)"
        fi
    } > "$tally"
    log "step1: final run-23 tally recorded at $tally"
    state_set step1_tally "$tally"
    state_mark step1
}

# ---------------------------------------------------------------- step 2: merge gate
pytest_failures() {  # pytest_failures <rootdir> <logfile> — failing/erroring node ids on stdout
    ( cd "$1" && python3 -m pytest -q -rfE --tb=no \
          scripts/kernel_rnd/autokernel/loop/ scripts/kernel_rnd/autokernel/controller/ ) > "$2" 2>&1 || true
    grep -aE '^(FAILED|ERROR) ' "$2" | awk '{print $1, $2}' | sort -u
}

run_gate_suite() {  # in the LANE, post-merge; floors/guards absolute, pytest is a DELTA gate
    local lf="$1" rc=0
    ( cd "$LANE_ROOT" && heavy suite_floors    "$lf" python3 scripts/kernel_rnd/autokernel/check_suite_floors.py )    || { log "step2: check_suite_floors.py RED"; rc=1; }
    ( cd "$LANE_ROOT" && heavy regrowth_guards "$lf" python3 scripts/kernel_rnd/autokernel/check_regrowth_guards.py ) || { log "step2: check_regrowth_guards.py RED"; rc=1; }
    # The controller/ scope carries pre-existing legacy red the rung payload does not own
    # (76F+10E, byte-identical failing sets measured at 74b936b5 and 9c40429d, 2026-09-02;
    # operator approved the delta gate that morning), so an absolute-green pytest gate here
    # can never pass. Refuse only on failures the merge INTRODUCES: baseline = the pre-merge
    # tag in a throwaway detached worktree.
    local base_fail="$WORK_DIR/step2-pytest-baseline-failures.txt"
    local merged_fail="$WORK_DIR/step2-pytest-merged-failures.txt"
    local basewt="$WORK_DIR/step2-baseline-wt" new_fail
    lane_git worktree remove --force "$basewt" >> "$lf" 2>&1 || true
    lane_git worktree add --detach "$basewt" "$PRE_MERGE_TAG" >> "$lf" 2>&1 \
        || { log "step2: could not create baseline worktree at $PRE_MERGE_TAG"; return 1; }
    pytest_failures "$basewt"    "$WORK_DIR/step2-pytest-baseline.log" > "$base_fail"
    lane_git worktree remove --force "$basewt" >> "$lf" 2>&1 || true
    pytest_failures "$LANE_ROOT" "$WORK_DIR/step2-pytest-merged.log"   > "$merged_fail"
    grep -qaE '[0-9]+ passed' "$WORK_DIR/step2-pytest-baseline.log" \
        || { log "step2: baseline pytest produced no pass count — collection broke; refusing"; rc=1; }
    grep -qaE '[0-9]+ passed' "$WORK_DIR/step2-pytest-merged.log" \
        || { log "step2: merged pytest produced no pass count — collection broke; refusing"; rc=1; }
    new_fail=$(comm -13 "$base_fail" "$merged_fail")
    if [[ -n "$new_fail" ]]; then
        log "step2: pytest delta gate RED — failures introduced by the merge (first 20):"
        printf '%s\n' "$new_fail" | head -20 | while IFS= read -r ln; do log "step2:   $ln"; done
        rc=1
    else
        log "step2: pytest delta gate green — $(wc -l < "$merged_fail") pre-existing failures (baseline $(wc -l < "$base_fail")), 0 introduced"
    fi
    return $rc
}

step2_merge_gate() {
    if state_done step2; then log "step2: already done, skipping"; return 0; fi
    local lf="$WORK_DIR/step2-merge.log" rung_commits lane_tip merged_tip
    log "step2: merge gate (log: $lf)"

    # Gate (a): the rung branch exists and carries >=1 reviewed-shape commit.
    lane_git rev-parse --verify -q "refs/heads/$RUNG_BRANCH" > /dev/null \
        || refuse step2 "branch $RUNG_BRANCH does not exist in the shared clone"
    rung_commits=$(lane_git log --format=%s "$LANE_BRANCH..$RUNG_BRANCH" | grep -c "^$RUNG_MSG_PREFIX") || true
    (( rung_commits >= 1 )) || refuse step2 "no commit in $LANE_BRANCH..$RUNG_BRANCH has a subject starting '$RUNG_MSG_PREFIX' (found $rung_commits)"
    state_set step2_rung_commits "$rung_commits"

    # Gate (b): the human-session review token.
    [[ -f "$REVIEW_TOKEN" ]] || refuse step2 "review token $REVIEW_TOKEN is missing — the owning session creates it only after human review of the rung commits; without it the merge must not happen"

    # Lane sanity: right branch, no tracked modifications (untracked files don't block a merge).
    [[ "$(lane_git branch --show-current)" == "$LANE_BRANCH" ]] \
        || refuse step2 "lane worktree $LANE_ROOT is on '$(lane_git branch --show-current)', not $LANE_BRANCH"
    [[ -z "$(lane_git status --porcelain --untracked-files=no)" ]] \
        || refuse step2 "lane worktree $LANE_ROOT has tracked modifications — refusing to merge into a dirty tree: $(lane_git status --porcelain --untracked-files=no | head -5 | tr '\n' ' ')"

    lane_tip=$(lane_git rev-parse HEAD)
    state_set lane_tip_pre_merge "$lane_tip"
    if lane_git rev-parse --verify -q "refs/tags/$PRE_MERGE_TAG" > /dev/null; then
        [[ "$(lane_git rev-parse "$PRE_MERGE_TAG")" == "$lane_tip" ]] \
            || refuse step2 "tag $PRE_MERGE_TAG already exists and is NOT the current lane tip — a previous merge attempt left state behind; operator review needed"
        log "step2: rollback tag $PRE_MERGE_TAG already at lane tip (resume)"
    else
        lane_git tag "$PRE_MERGE_TAG" "$lane_tip"
        log "step2: rollback tag $PRE_MERGE_TAG = ${lane_tip:0:12}"
    fi

    if ! heavy lane_merge "$lf" \
            git -C "$LANE_ROOT" merge --no-ff "$RUNG_BRANCH" \
            -m "autokernel boundary: merge $RUNG_BRANCH at the run-23→24 boundary (loop verified dead)"; then
        lane_git merge --abort >> "$lf" 2>&1 || true
        lane_git reset --hard "$PRE_MERGE_TAG" >> "$lf" 2>&1
        refuse step2 "merge of $RUNG_BRANCH into $LANE_BRANCH failed (see $lf) — lane rolled back to $PRE_MERGE_TAG"
    fi
    merged_tip=$(lane_git rev-parse HEAD)
    state_set lane_tip_post_merge "$merged_tip"
    log "step2: merged — lane at ${merged_tip:0:12}"

    if ! run_gate_suite "$lf"; then
        lane_git reset --hard "$PRE_MERGE_TAG" >> "$lf" 2>&1
        refuse step2 "post-merge gate suite RED (see $lf) — lane rolled back to $PRE_MERGE_TAG; the merge is refused"
    fi
    log "step2: gate suite green (check_suite_floors, check_regrowth_guards, pytest loop/+controller/)"
    state_mark step2
}

# ---------------------------------------------------------------- step 3: seeds
step3_seeds() {
    if state_done step3; then log "step3: already done, skipping"; return 0; fi
    local f name copied=0 manifest="$WORK_DIR/step3-seeds-manifest.txt"
    log "step3: staged re-anchored seeds -> $INBOX"

    [[ -d "$SEED_STAGING" ]] || refuse step3 "seed staging dir $SEED_STAGING is missing"
    compgen -G "$SEED_STAGING/*" > /dev/null || refuse step3 "seed staging dir $SEED_STAGING is empty"
    [[ -d "$INBOX" ]] || refuse step3 "store inbox $INBOX is missing"
    mkdir -p "$INBOX_BACKUP"

    : > "$manifest"
    for f in "$SEED_STAGING"/*; do
        [[ -f "$f" ]] || continue
        name=$(basename "$f")
        if [[ -f "$INBOX/$name" ]]; then
            cp -p "$INBOX/$name" "$INBOX_BACKUP/$name"
            echo "replaced $name (original backed up: $(sha256sum "$INBOX_BACKUP/$name" | cut -d' ' -f1))" >> "$manifest"
        else
            echo "new      $name" >> "$manifest"
        fi
        cp -p "$f" "$INBOX/$name"
        echo "  staged -> inbox: $(sha256sum "$INBOX/$name" | cut -d' ' -f1)" >> "$manifest"
        copied=$(( copied + 1 ))
    done
    (( copied >= 1 )) || refuse step3 "seed staging dir $SEED_STAGING contains no regular files"
    log "step3: $copied seed file(s) copied (manifest: $manifest)"
    state_set step3_copied "$copied"
    state_mark step3
}

# ---------------------------------------------------------------- step 4: rocprof identity
verify_frozen_production() {
    local br head
    br=$(git -C "$PROD_TREE" branch --show-current)
    head=$(git -C "$PROD_TREE" rev-parse HEAD)
    [[ "$br" == "$PROD_BRANCH" && "$head" == "$PROD_COMMIT" ]] \
        || refuse step4 "frozen production tree is $br@${head:0:12}, expected $PROD_BRANCH@${PROD_COMMIT:0:12} — do NOT touch it; escalate"
    [[ -x "$PROD_BENCH" ]] || refuse step4 "frozen production llama-bench missing at $PROD_BENCH (this driver never builds production)"
}

step4_rocprof() {
    if state_done step4; then log "step4: already done, skipping"; return 0; fi
    local surface lf out
    log "step4: rocprofv3 dispatch sanity on the FROZEN production build (reference dispatch table)"
    verify_frozen_production
    [[ -f "$MODEL_27B" ]] || refuse step4 "27B production model missing at $MODEL_27B"
    mkdir -p "$RUNG_IDENTITY_DIR"

    for surface in "${CONFIRM_SURFACES[@]}"; do
        if state_done "step4_$surface"; then log "step4: $surface already done, skipping"; continue; fi
        lf="$WORK_DIR/step4-rocprof-$surface.log"
        out="$RUNG_IDENTITY_DIR/$surface.$MODEL_27B_STEM.json"
        log "step4: $surface kernel-trace (log: $lf)"
        heavy "rocprof_$surface" "$lf" \
            python3 "$HELPER" rocprof \
                --lane-root "$LANE_ROOT" --binary "$PROD_BENCH" \
                --model "$MODEL_27B" --surface "$surface" --out "$out" \
            || refuse step4 "$surface rocprof dispatch capture failed (see $lf)"
        [[ -s "$out" ]] || refuse step4 "$surface dispatch table $out was not written"
        log "step4: $surface dispatch table -> $out"
        state_mark "step4_$surface"
    done
    state_mark step4
}

# ---------------------------------------------------------------- step 5: DFlash2 smoke
step5_dflash_smoke() {   # informs D5; failure is RECORDED and the boundary CONTINUES
    if state_done step5; then log "step5: already done, skipping"; return 0; fi
    local lf="$WORK_DIR/step5-dflash-smoke.log" out="$WORK_DIR/step5-dflash-smoke.json" rc=0
    log "step5: DFlash2 drafter-head smoke (log: $lf) — non-gating"
    if [[ ! -f "$DFLASH2_GGUF" ]]; then
        log "step5: DFlash2 GGUF missing at $DFLASH2_GGUF — recorded, continuing"
        state_set step5_outcome "model_missing"
        state_mark step5; return 0
    fi
    heavy dflash_smoke "$lf" \
        python3 "$HELPER" smoke \
            --lane-root "$LANE_ROOT" --binary "$PROD_BENCH" \
            --model "$DFLASH2_GGUF" --out "$out" || rc=$?
    if [[ $rc -eq 0 ]]; then
        log "step5: smoke PASSED ($out)"
        state_set step5_outcome "passed"
    else
        log "step5: smoke FAILED rc=$rc (see $lf) — recorded, boundary CONTINUES (informs D5, does not gate)"
        state_set step5_outcome "failed_rc$rc"
    fi
    state_mark step5
}

# ---------------------------------------------------------------- step 6: 27B A/A calibration
step6_calibration() {
    local surface lf anchor rc calib
    anchor="${BOUNDARY_CALIB_ANCHOR:-$(latest_anchor_gen)}"
    [[ -n "$anchor" ]] || refuse step6 "no anchor-gen-* with provenance.json in $STORE (and BOUNDARY_CALIB_ANCHOR unset) — nothing to calibrate against"
    [[ -f "$anchor/provenance.json" ]] || refuse step6 "calibration anchor $anchor carries no provenance.json"
    state_set step6_anchor "$anchor"
    log "step6: 27B A/A calibration, anchor $anchor, $CALIB_PAIRS pairs/surface (POST-MERGE lane code; long step — output to files only)"

    for surface in "${CONFIRM_SURFACES[@]}"; do
        calib="$STORE/calibration/$surface.$MODEL_27B_STEM.json"
        if state_done "step6_$surface"; then log "step6: $surface already done, skipping"; continue; fi
        lf="$WORK_DIR/step6-calib-$surface.log"
        log "step6: calibrating $surface on $MODEL_27B_STEM (log: $lf)"
        rc=0
        ( cd "$LANE_ROOT/scripts/kernel_rnd" && heavy "calibrate_$surface" "$lf" \
              python3 -u -m autokernel.loop.run \
              --worktree "$CHAMP_TREE" --anchor-build "$anchor" \
              --model "$MODEL_27B" --store "$STORE" \
              --surface "$surface" --calibrate-surface "$CALIB_PAIRS" ) || rc=$?
        [[ $rc -eq 0 && -f "$calib" ]] \
            || refuse step6 "$surface calibration failed (rc=$rc, keyed record $calib $([[ -f "$calib" ]] && echo present || echo MISSING); see $lf)"
        log "step6: $surface keyed floor written -> $calib"
        state_mark "step6_$surface"
    done
    state_mark step6
}

# ------------------------------------------------------- all-green conjunction
all_green_reasons() {   # one reason per line; EMPTY output == all green
    local surface outcome
    outcome=$(state_get step1_outcome)
    if ! state_done step1 || [[ "$outcome" != "stopped" && "$outcome" != "already_dead" ]]; then
        echo "step1: run 23 not verified stopped (outcome=${outcome:-none})"
    fi
    state_done step2 || echo "step2: merge gate not green (rung branch merged + gate suite)"
    state_done step3 || echo "step3: re-anchored seeds not placed in the inbox"
    for surface in "${CONFIRM_SURFACES[@]}"; do
        state_done "step4_$surface" || echo "step4: $surface rung-identity dispatch table missing"
        if ! state_done "step6_$surface" || [[ ! -f "$STORE/calibration/$surface.$MODEL_27B_STEM.json" ]]; then
            echo "step6: $surface keyed 27B calibration missing"
        fi
    done
    # step5 is non-gating by design; its outcome is reported, never a red reason.
}

evaluate_all_green() {
    local reasons
    reasons=$(all_green_reasons)
    if [[ -z "$reasons" ]]; then
        state_set allgreen "yes"
        log "all-green: YES — every gating step green"
    else
        state_set allgreen "no"
        while IFS= read -r line; do
            state_set allgreen_reason "$line"
            log "all-green: NO — $line"
        done <<< "$reasons"
    fi
}

# ---------------------------------------------------------------- step 7: readiness (+ preauth launch)
run24_argv_doc() {  # the proposed run-24 command, single source for report + launch
    # Anchor = the loop's own latest guard-verified anchor-gen (A/A'd against the
    # champion by the run-23 advance chain), NEVER an incremental tip rebuild under
    # --allow-unverified-anchor: the incremental champ2 build is the exact pattern
    # that failed digest attestation on 2026-08-31, and the waiver would mask it.
    # If the tip moved after the last anchor promotion (a 21:59Z keep), the loop's
    # startup refusal correctly HOLDS run 24 for morning — fail-closed, by design.
    local anchor; anchor="${BOUNDARY_RUN24_ANCHOR:-$(latest_anchor_gen)}"
    cat <<EOF
cd $LANE_ROOT/scripts/kernel_rnd && \\
  setsid nohup python3 -u -m autokernel.loop.run \\
      --worktree $CHAMP_TREE \\
      --anchor-build ${anchor:-<no provenance-carrying anchor-gen in $STORE>} \\
      --model $MODEL_SCREEN \\
      --confirm-model $MODEL_27B --confirm-pairs 5 --confirm-surfaces dec-b4,dec-b8 \\
      --store $STORE \\
      --iterations 0 --surface dec-b4 --pairs 20 --workers 7 \\
      --rank-prior-experiments \\
      --worker-root /mnt/raid0/llm/tmp/ak-lanes \\
      --worker-build-root /mnt/raid0/llm/tmp/ak-lane-builds \\
      --out $RUN24_OUT \\
      > $RUN24_LOG 2>&1 < /dev/null &
echo \$! > $RUN24_PIDFILE
EOF
}

write_readiness() {
    python3 "$HELPER" report \
        --state "$STATE_FILE" --work-dir "$WORK_DIR" --store "$STORE" \
        --lane-root "$LANE_ROOT" --model27 "$MODEL_27B" \
        --readiness "$READINESS" --run24-cmd-file "$WORK_DIR/run24-command.sh" \
        >> "$WORK_DIR/step7-report.log" 2>&1 \
        || log "step7: WARNING report writer failed (see $WORK_DIR/step7-report.log)"
    log "step7: readiness package -> $READINESS"
}

launch_run24() {  # the 20260831 watcher mechanism, folded in; call ONLY preauth+allgreen
    local lf="$WORK_DIR/step7-launch.log" dry rc p cl found=""
    log "step7: PREAUTH_RUN24 present + all green — launching run 24 (watcher mechanism)"

    # Anchor for run 24: the loop's own latest guard-verified anchor-gen — see the
    # rationale in run24_argv_doc. No tip rebuild, no --allow-unverified-anchor.
    local anchor; anchor="${BOUNDARY_RUN24_ANCHOR:-$(latest_anchor_gen)}"
    if [[ -z "$anchor" || ! -x "$anchor/bin/llama-bench" ]]; then
        state_set run24_launch "held_no_verified_anchor"
        log "step7: HOLD — no provenance-carrying anchor-gen with bin/llama-bench in $STORE; run 24 NOT started"; return 0
    fi

    # Final refusal check: the loop's own --dry-run must exit 0 AND print "— verified".
    # No waiver flag: the startup refusal (anchor attachment == champion tip) must
    # itself pass, or the launch holds for morning with the refusal in the log.
    rc=0
    dry=$( cd "$LANE_ROOT/scripts/kernel_rnd" && python3 -m autokernel.loop.run \
            --worktree "$CHAMP_TREE" --anchor-build "$anchor" \
            --model "$MODEL_SCREEN" \
            --confirm-model "$MODEL_27B" --confirm-pairs 5 --confirm-surfaces dec-b4,dec-b8 \
            --store "$STORE" --iterations 0 --surface dec-b4 --pairs 20 --dry-run 2>&1 ) || rc=$?
    echo "$dry" >> "$lf"
    if [[ $rc -ne 0 ]] || ! grep -q "— verified" <<< "$dry"; then
        state_set run24_launch "held_dry_run_refused_rc$rc"
        log "step7: HOLD — run-24 dry-run refusal check failed (rc=$rc, see $lf); NOT started"; return 0
    fi

    ( cd "$LANE_ROOT/scripts/kernel_rnd" && \
      setsid nohup python3 -u -m autokernel.loop.run \
          --worktree "$CHAMP_TREE" \
          --anchor-build "$anchor" \
          --model "$MODEL_SCREEN" \
          --confirm-model "$MODEL_27B" --confirm-pairs 5 --confirm-surfaces dec-b4,dec-b8 \
          --store "$STORE" \
          --iterations 0 --surface dec-b4 --pairs 20 --workers 7 \
          --rank-prior-experiments \
          --worker-root /mnt/raid0/llm/tmp/ak-lanes \
          --worker-build-root /mnt/raid0/llm/tmp/ak-lane-builds \
          --out "$RUN24_OUT" > "$RUN24_LOG" 2>&1 < /dev/null & )

    sleep 20
    for p in /proc/[0-9]*; do            # read-only /proc scan (watcher pattern), never a kill
        [[ -r "$p/cmdline" ]] || continue
        cl=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
        case "$cl" in *"autokernel.loop.run"*run24*) found=${p#/proc/} ;; esac
    done
    if [[ -n "$found" ]]; then
        echo "$found" > "$RUN24_PIDFILE"
        state_set run24_launch "launched_pid_$found"
        log "step7: run 24 LAUNCHED pid=$found (pid file $RUN24_PIDFILE)"
        head -8 "$RUN24_LOG" >> "$lf" || true
    else
        state_set run24_launch "launch_failed_no_process"
        log "step7: LAUNCH FAILED — no run24 process found after 20s (see $RUN24_LOG)"
    fi
}

step7_readiness() {
    log "step7: readiness package (no device)"
    evaluate_all_green
    run24_argv_doc > "$WORK_DIR/run24-command.sh"
    write_readiness
    if [[ -f "$PREAUTH_TOKEN" && "$(state_get allgreen)" == "yes" ]]; then
        launch_run24
        write_readiness   # re-write so the package records the launch outcome
    else
        if [[ -f "$PREAUTH_TOKEN" ]]; then
            log "step7: PREAUTH_RUN24 present but NOT all green — run 24 NOT started (reasons in $READINESS)"
            state_set run24_launch "held_not_all_green"
        else
            log "step7: no PREAUTH_RUN24 token — stopping at the package; run 24 NOT started"
            state_set run24_launch "no_preauth_token"
        fi
        write_readiness   # re-write so the package records the held/absent state
    fi
    state_mark step7
}

# ---------------------------------------------------------------- dry run
check() {  # check <label> <ok?0:1> <detail>
    local label="$1" ok="$2" detail="$3"
    if [[ "$ok" == 0 ]]; then printf '  [ SAT ] %-34s %s\n' "$label" "$detail"
    else printf '  [UNSAT] %-34s %s\n' "$label" "$detail"; fi
}

dry_run_plan() {
    local pid="?" alive="n/a" cl="" rung_ok=1 rung_n=0 anchor prod_head prod_br lane_br
    echo "================ boundary_20260901 DRY RUN — nothing will be executed ================"
    echo "work dir   $WORK_DIR   (state: $STATE_FILE)   T0: ${T0_UTC}Z"
    echo
    echo "-- preconditions checkable now (UNSAT = would refuse tonight unless it appears) --"
    date -u -d "$T0_UTC" +%s > /dev/null 2>&1 && check "T0 parseable" 0 "${T0_UTC}Z" || check "T0 parseable" 1 "'$T0_UTC'"
    if [[ -f "$PID_FILE" ]]; then
        pid=$(tr -dc '0-9' < "$PID_FILE"); cl=$(read_cmdline "$pid")
        if pid_alive "$pid"; then
            loop23_cmdline_ok "$pid" && alive="ALIVE, cmdline verified ('$LOOP_MATCH_A' + '$LOOP_MATCH_B')" \
                                     || alive="ALIVE but cmdline MISMATCH (would REFUSE): ${cl:0:120}"
        else alive="dead (verifies clean as already_dead)"; fi
        [[ "$alive" == *MISMATCH* ]] && check "run23 pid" 1 "pid $pid — $alive" || check "run23 pid" 0 "pid $pid — $alive"
    else
        check "run23 pid file" 1 "$PID_FILE missing (step1 would REFUSE)"
    fi
    lane_br=$(lane_git branch --show-current 2>/dev/null || echo "?")
    [[ "$lane_br" == "$LANE_BRANCH" ]] && check "lane branch" 0 "$LANE_ROOT on $lane_br" || check "lane branch" 1 "$LANE_ROOT on '$lane_br', need $LANE_BRANCH"
    [[ -z "$(lane_git status --porcelain --untracked-files=no 2>/dev/null)" ]] && check "lane tree clean (tracked)" 0 "" || check "lane tree clean (tracked)" 1 "tracked modifications present (may be the other agent's in-flight work)"
    if lane_git rev-parse --verify -q "refs/heads/$RUNG_BRANCH" > /dev/null 2>&1; then rung_ok=0
        rung_n=$(lane_git log --format=%s "$LANE_BRANCH..$RUNG_BRANCH" 2>/dev/null | grep -c "^$RUNG_MSG_PREFIX" || true); fi
    check "rung branch" $rung_ok "$RUNG_BRANCH ($rung_n '$RUNG_MSG_PREFIX' commit(s) ahead of lane)"
    (( rung_n >= 1 )) && check "rung commit prefix" 0 "$rung_n commit(s)" || check "rung commit prefix" 1 "0 commits with subject '$RUNG_MSG_PREFIX' in $LANE_BRANCH..$RUNG_BRANCH"
    [[ -f "$REVIEW_TOKEN" ]] && check "review token" 0 "$REVIEW_TOKEN" || check "review token" 1 "$REVIEW_TOKEN absent (owning session writes it after human review)"
    lane_git rev-parse --verify -q "refs/tags/$PRE_MERGE_TAG" > /dev/null 2>&1 && check "rollback tag unused" 1 "$PRE_MERGE_TAG already exists (stale merge attempt?)" || check "rollback tag unused" 0 "$PRE_MERGE_TAG free"
    if [[ -d "$SEED_STAGING" ]] && compgen -G "$SEED_STAGING/*" > /dev/null; then
        check "seed staging" 0 "$SEED_STAGING ($(ls "$SEED_STAGING" | wc -l) file(s))"
    else check "seed staging" 1 "$SEED_STAGING missing or empty"; fi
    [[ -d "$INBOX" ]] && check "store inbox" 0 "$INBOX" || check "store inbox" 1 "$INBOX missing"
    prod_br=$(git -C "$PROD_TREE" branch --show-current 2>/dev/null || echo "?")
    prod_head=$(git -C "$PROD_TREE" rev-parse HEAD 2>/dev/null || echo "?")
    [[ "$prod_br" == "$PROD_BRANCH" && "$prod_head" == "$PROD_COMMIT" ]] \
        && check "frozen production tree" 0 "$prod_br@${prod_head:0:12}" \
        || check "frozen production tree" 1 "$prod_br@${prod_head:0:12}, need $PROD_BRANCH@${PROD_COMMIT:0:12}"
    [[ -x "$PROD_BENCH" ]] && check "production llama-bench" 0 "$PROD_BENCH" || check "production llama-bench" 1 "$PROD_BENCH missing"
    python3 "$HELPER" probe --lane-root "$LANE_ROOT" 2>/dev/null | sed 's/^/  /' \
        || echo "  [UNSAT] helper probe                      $HELPER probe failed (rocprofv3/claim import)"
    [[ -f "$MODEL_27B" ]] && check "27B model" 0 "$MODEL_27B" || check "27B model" 1 "$MODEL_27B missing"
    [[ -f "$MODEL_SCREEN" ]] && check "screen model (1.5B)" 0 "$MODEL_SCREEN" || check "screen model (1.5B)" 1 "$MODEL_SCREEN missing"
    [[ -f "$DFLASH2_GGUF" ]] && check "DFlash2 drafter GGUF" 0 "$DFLASH2_GGUF (non-gating)" || check "DFlash2 drafter GGUF" 1 "$DFLASH2_GGUF missing (non-gating: recorded+continue)"
    [[ -x "$CHAMP_TREE/build-hip/bin/llama-bench" ]] && check "champion worktree/build" 0 "$CHAMP_TREE/build-hip" || check "champion worktree/build" 1 "$CHAMP_TREE/build-hip/bin/llama-bench missing"
    anchor="${BOUNDARY_CALIB_ANCHOR:-$(latest_anchor_gen || true)}"
    [[ -n "$anchor" && -f "$anchor/provenance.json" ]] && check "calibration anchor" 0 "$anchor (latest provenance-carrying anchor-gen; re-resolved at run time)" \
        || check "calibration anchor" 1 "no anchor-gen-* with provenance.json under $STORE"
    [[ -f "$PREAUTH_TOKEN" ]] && check "PREAUTH_RUN24" 0 "present — step7 WOULD launch run 24 if all green" || check "PREAUTH_RUN24" 1 "absent — step7 stops at the readiness package (this is the default posture, not an error)"
    ( cd "$LANE_ROOT/scripts/kernel_rnd" 2>/dev/null && python3 -c "import autokernel.loop.run" 2>/dev/null ) \
        && check "lane loop importable" 0 "pre-merge lane code imports (confirm-rung flags land WITH the step-2 merge)" \
        || check "lane loop importable" 1 "autokernel.loop.run does not import from $LANE_ROOT/scripts/kernel_rnd"
    echo
    cat <<EOF
-- plan --
step0  wait until ${T0_UTC}Z (epoch pinned in state on first run; resumable)
step1  SIGTERM pid from $PID_FILE (cmdline must contain '$LOOP_MATCH_A' AND '$LOOP_MATCH_B');
       ${KILL_WAIT_S}s grace -> SIGKILL group-of-pid -> verify /proc gone; PGID remnant sweep
       (recorded pid only); record run-23 tally tail + loop-status.json dispositions
step2  gates: >=1 '$RUNG_MSG_PREFIX' commit in $LANE_BRANCH..$RUNG_BRANCH AND $REVIEW_TOKEN;
       tag $PRE_MERGE_TAG; git merge --no-ff $RUNG_BRANCH IN $LANE_ROOT;
       then check_suite_floors.py + check_regrowth_guards.py + pytest loop/ controller/;
       any red => reset --hard to the tag + REFUSE
step3  cp $SEED_STAGING/* -> $INBOX (replaced originals -> $INBOX_BACKUP; sha256 manifest)
step4  helper rocprof per surface (${CONFIRM_SURFACES[*]}): rocprofv3 --kernel-trace on
       $PROD_BENCH, $MODEL_27B_STEM, dec-b4=-p512 -n0 -b4 -ub4 / dec-b8=-b8 -ub8;
       claim.hold(mi210_0) for each window; tables -> $RUNG_IDENTITY_DIR/
step5  helper smoke on $DFLASH2_GGUF (tg128 + dec-b4 shapes); failure recorded, NOT gating
step6  per surface: cd $LANE_ROOT/scripts/kernel_rnd && python3 -u -m autokernel.loop.run
       --worktree $CHAMP_TREE --anchor-build <latest anchor-gen> --model $MODEL_27B
       --store $STORE --surface <s> --calibrate-surface $CALIB_PAIRS
       => $STORE/calibration/<s>.$MODEL_27B_STEM.json (keyed floors; ~5-6h total)
step7  $READINESS + run-24 command file; launch ONLY if $PREAUTH_TOKEN exists AND all
       gating steps green (latest verified anchor-gen -> loop --dry-run '— verified' -> setsid launch
       -> read-only /proc scan -> $RUN24_PIDFILE); otherwise stop at the package

-- proposed run-24 command (two-rung: screen 1.5B / confirm 27B) --
$(run24_argv_doc)
=====================================================================================
EOF
}

# ---------------------------------------------------------------- main
main() {
    local now_flag=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --now)     now_flag=1; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            *) echo "usage: boundary_20260901.sh [--now] [--dry-run]" >&2; return 2 ;;
        esac
    done

    if [[ "$DRY_RUN" == 1 ]]; then dry_run_plan; return 0; fi

    mkdir -p "$WORK_DIR"
    touch "$STATE_FILE"
    log "=== boundary run-23->24 driver start (state: $STATE_FILE) ==="
    if [[ "$now_flag" == 1 ]]; then state_done step0 || state_mark step0; log "step0: skipped (--now)"; fi

    step0_wait
    step1_stop_run23
    step2_merge_gate
    step3_seeds
    step4_rocprof
    step5_dflash_smoke
    step6_calibration
    step7_readiness
    state_set driver_done 1
    log "=== boundary driver complete — see $READINESS ==="
}

if [[ "${BOUNDARY_LIB_ONLY:-0}" != 1 ]]; then
    main "$@"
    exit $?
fi
