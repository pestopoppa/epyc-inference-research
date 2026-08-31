#!/bin/bash
# boundary_20260831.sh — unattended driver for the AutoKernel run-21→22 boundary.
#
#   boundary_20260831.sh --at 22:00     # sleep until 22:00Z today, then run
#   boundary_20260831.sh --now          # run immediately
#   boundary_20260831.sh --dry-run      # print the fully resolved plan, execute nothing
#
# Encodes the boundary checklist from FUNSAFE_MATH_ADMISSION_NOTE.md, adapted for
# unattended operation (operator scheduling ruling, 2026-08-31: "run the boundary
# overnight starting at 22:00Z"):
#
#   step0  stop run 21          STOP sentinel + SIGTERM; T0+25min SIGKILL the group
#   step1  funsafe-math A/B     autokernel_funsafe_math_admission.py; merge the
#                               flag removal onto the champion ONLY on measured
#                               greedy DIVERGENCE (operator ruling: "the 2% decode
#                               hit is worth it IF it increases quality as stated")
#   step2  dec-b2/b4/b8 floors  autokernel.loop.run --calibrate-surface (D8 method)
#   step3  serving refresh      serving_evidence_refresh.py, full grid
#   step4  readiness report     /mnt/raid0/llm/tmp/boundary-20260831-report.md
#
# Run 22 is NEVER launched by this driver — run starts stay operator-gated. The
# operator ruling of 2026-08-31 (OP-32: "pre-authorize run 22 if all boundary
# steps are green") is honoured by EVALUATING its exact all-green conjunction
# and reporting the verdict with every red reason; the launch command is in the
# report, ready to paste. (Encoding the unattended start itself was denied by
# the permission system at build time; the default on any doubt is HOLD.)
#
# Failure policy: set -uo pipefail but NOT -e. Each step is wrapped; a failed
# step 1-3 is logged and stepped past (per-step rules below); only an unkillable
# run 21 aborts the device steps (everything after it needs the claim free).
#
# Idempotent-ish: $WORK_DIR/state records completed steps; crash+relaunch skips
# done work. Delete the state file to force a full re-run.
#
# Claim discipline: each harness holds the mi210_0 claim itself (claim.hold);
# this driver NEVER touches /mnt/raid0/llm/tmp/gpu_device.mi210_0.lock.
# Kill discipline: signals go only to the pid self-read from run21.pid (and its
# process group, resolved from that pid); never a name-pattern kill.
#
# Test seams (all default to production values):
#   BOUNDARY_STUB_DIR         dir of executable stubs replacing heavy commands
#   BOUNDARY_WORK_DIR / BOUNDARY_PID_FILE / BOUNDARY_STORE / BOUNDARY_CHAMP_TREE
#   BOUNDARY_ADMISSION_TREE / BOUNDARY_SURFACE_DIR / BOUNDARY_REPORT
#   BOUNDARY_KILL_WAIT_S / BOUNDARY_KILL_GRACE_S / BOUNDARY_POLL_S
#   BOUNDARY_LOOP_MATCH       substring that must appear in the pid's cmdline
#   BOUNDARY_FAKE_NOW         epoch override for the --at wait computation
#   BOUNDARY_LIB_ONLY=1       source the file for its functions, run nothing
set -uo pipefail

# ---------------------------------------------------------------- configuration
LANE_ROOT="${BOUNDARY_LANE_ROOT:-/mnt/raid0/llm/worktrees/mains/ak-rebuild-research}"
BENCH_DIR="$LANE_ROOT/scripts/benchmark"
KERNEL_RND="$LANE_ROOT/scripts/kernel_rnd"
HELPER="$BENCH_DIR/boundary_20260831_helper.py"

WORK_DIR="${BOUNDARY_WORK_DIR:-/mnt/raid0/llm/tmp/boundary-20260831}"
STATE_FILE="$WORK_DIR/state"
PID_FILE="${BOUNDARY_PID_FILE:-/mnt/raid0/llm/tmp/run21.pid}"
STORE="${BOUNDARY_STORE:-/mnt/raid0/llm/autokernel/loop-memory}"
STOP_SENTINEL="$STORE/STOP"
SURFACE_DIR="${BOUNDARY_SURFACE_DIR:-/mnt/raid0/llm/autokernel/surface}"
REPORT="${BOUNDARY_REPORT:-/mnt/raid0/llm/tmp/boundary-20260831-report.md}"

CHAMP_TREE="${BOUNDARY_CHAMP_TREE:-/mnt/raid0/llm/tmp/champ2}"
CHAMPION_BRANCH="${BOUNDARY_CHAMPION_BRANCH:-ak/champion/llama-cpp-0db32c06e3e5}"
ADMISSION_TREE="${BOUNDARY_ADMISSION_TREE:-/mnt/raid0/llm/tmp/ak-admission-funsafe-20260831}"
ADMISSION_REF="${BOUNDARY_ADMISSION_REF:-ak/admission/remove-funsafe-math-20260831}"
PRE_MERGE_TAG="ak/pre-funsafe-merge-20260831"
MODEL="${BOUNDARY_MODEL:-/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf}"
ANCHOR_GEN="${BOUNDARY_ANCHOR_GEN:-$STORE/anchor-gen-005}"   # provenance: aba5a8155cdd
FUNSAFE_BUILD_ROOT="${BOUNDARY_FUNSAFE_BUILD_ROOT:-/mnt/raid0/llm/tmp/funsafe-admission-builds-boundary20260831}"
FUNSAFE_OUT="$WORK_DIR/funsafe-math-admission"
FUNSAFE_JSON="$FUNSAFE_OUT/funsafe-math-admission.json"

KILL_WAIT_S="${BOUNDARY_KILL_WAIT_S:-1500}"    # T0+25 min before SIGKILL (pre-drain-tier run 21)
KILL_GRACE_S="${BOUNDARY_KILL_GRACE_S:-60}"    # wait after SIGKILL before declaring failure
POLL_S="${BOUNDARY_POLL_S:-30}"
LOOP_MATCH="${BOUNDARY_LOOP_MATCH:-autokernel.loop}"
CALIB_PAIRS="${BOUNDARY_CALIB_PAIRS:-20}"      # aa_campaign default; run-21 measured at 20 pairs
CALIB_SURFACES=(dec-b2 dec-b4 dec-b8)

# ROCm toolchain env, mirrored from run 21's live environment (read 2026-08-31).
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-/opt/rocm}"
case ":$PATH:" in *":/opt/rocm/bin:"*) ;; *) export PATH="/opt/rocm/bin:$PATH" ;; esac
export LD_LIBRARY_PATH="/opt/AMD/aocc-compiler-5.0.0/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DRY_RUN=0

# ---------------------------------------------------------------- plumbing
log() {   # log <msg...>  — timestamped, to driver log and stdout
    local line
    line="[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
    echo "$line"
    [[ "$DRY_RUN" == 1 ]] || echo "$line" >> "$WORK_DIR/driver.log"
}

state_done() { [[ -f "$STATE_FILE" ]] && grep -q "^$1=done$" "$STATE_FILE"; }
state_mark() { echo "$1=done" >> "$STATE_FILE"; }
state_set()  { echo "$1=$2" >> "$STATE_FILE"; }
state_get()  { [[ -f "$STATE_FILE" ]] && grep "^$1=" "$STATE_FILE" | tail -1 | cut -d= -f2- || true; }

heavy() {  # heavy <stub-name> <logfile> <argv...>  — the stub seam for testing
    local name="$1" lf="$2"; shift 2
    if [[ -n "${BOUNDARY_STUB_DIR:-}" && -x "${BOUNDARY_STUB_DIR}/$name" ]]; then
        "${BOUNDARY_STUB_DIR}/$name" "$@" >> "$lf" 2>&1
    else
        "$@" >> "$lf" 2>&1
    fi
}

seconds_until_utc() {  # seconds_until_utc HH:MM — 0 if already past today
    local target now
    target=$(date -u -d "today $1" +%s) || return 1
    now="${BOUNDARY_FAKE_NOW:-$(date -u +%s)}"
    if (( target <= now )); then echo 0; else echo $(( target - now )); fi
}

pid_alive() { [[ -d "/proc/$1" ]]; }

loop_alive() {  # loop_alive <pid> — alive AND cmdline matches LOOP_MATCH (pid-reuse guard)
    local pid="$1"
    pid_alive "$pid" || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qF "$LOOP_MATCH"
}

# ---------------------------------------------------------------- step 0: stop run 21
step0_stop_run21() {
    if state_done step0; then log "step0: already done, skipping"; return 0; fi
    local lf="$WORK_DIR/step0-stop.log" t0 pid pgid waited remnants
    log "step0: stopping run 21 (log: $lf)"

    if [[ ! -f "$PID_FILE" ]]; then
        log "step0: WARNING pid file $PID_FILE missing — assuming run 21 already stopped"
        rm -f "$STOP_SENTINEL"
        state_set step0_outcome "no_pid_file"
        state_mark step0
        return 0
    fi
    pid=$(tr -dc '0-9' < "$PID_FILE")
    if [[ -z "$pid" ]]; then
        log "step0: FATAL pid file $PID_FILE holds no pid"
        state_set step0_outcome "bad_pid_file"
        return 1
    fi
    state_set run21_pid "$pid"

    if ! loop_alive "$pid"; then
        log "step0: pid $pid is not a live '$LOOP_MATCH' process — run 21 already stopped"
        rm -f "$STOP_SENTINEL"
        state_set step0_outcome "already_dead"
        state_mark step0
        _log_loop_status "$lf"
        return 0
    fi

    # T0: STOP sentinel AND SIGTERM, per the boundary checklist.
    printf 'boundary_20260831.sh stop request %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STOP_SENTINEL"
    log "step0: wrote STOP sentinel $STOP_SENTINEL"
    kill -TERM "$pid" 2>> "$lf" && log "step0: SIGTERM sent to $pid" \
        || log "step0: WARNING SIGTERM to $pid failed (raced exit?)"

    # Run 21 is PRE-drain-tier (started before 95eeb0ae): the tail lane may spend
    # ~40 min in actor calls. Grace period, then group SIGKILL.
    waited=0
    while (( waited < KILL_WAIT_S )) && pid_alive "$pid"; do
        sleep "$POLL_S"; waited=$(( waited + POLL_S ))
    done

    if pid_alive "$pid"; then
        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -dc '0-9')
        if [[ -n "$pgid" && "$pgid" != "$(ps -o pgid= -p $$ | tr -dc '0-9')" ]]; then
            log "step0: still alive after ${KILL_WAIT_S}s — SIGKILL process group $pgid"
            kill -KILL -- "-$pgid" 2>> "$lf"
            state_set step0_escalated "sigkill_group_$pgid"
        else
            log "step0: WARNING pgid '$pgid' unusable or equals ours — SIGKILL pid only"
            kill -KILL "$pid" 2>> "$lf"
            state_set step0_escalated "sigkill_pid"
        fi
        waited=0
        while (( waited < KILL_GRACE_S )) && pid_alive "$pid"; do
            sleep "$POLL_S"; waited=$(( waited + POLL_S ))
        done
    else
        state_set step0_escalated "none_sigterm_sufficed"
    fi

    if pid_alive "$pid"; then
        log "step0: FATAL pid $pid survived SIGKILL — device steps cannot proceed"
        state_set step0_outcome "unkillable"
        return 1
    fi
    log "step0: pid $pid confirmed dead (/proc/$pid gone)"

    # Group remnants (llama-bench children etc.): observe by PGID of the self-read
    # pid — never by name pattern — and re-KILL the group if anything remains.
    pgid=$(state_get step0_escalated | grep -o '[0-9]*$' || true)
    pgid="${pgid:-$pid}"   # setsid'd loop: pgid == pid
    remnants=$(ps -eo pid=,pgid= | awk -v g="$pgid" '$2==g {print $1}')
    if [[ -n "$remnants" ]]; then
        log "step0: group $pgid remnants: $(echo "$remnants" | tr '\n' ' ')— SIGKILL group"
        kill -KILL -- "-$pgid" 2>> "$lf"
        sleep "$POLL_S"
        remnants=$(ps -eo pid=,pgid= | awk -v g="$pgid" '$2==g {print $1}')
        [[ -n "$remnants" ]] && log "step0: WARNING remnants persist: $remnants"
    fi
    state_set step0_group_remnants "${remnants:-none}"

    _check_gpu_idle "$lf"

    rm -f "$STOP_SENTINEL"
    log "step0: STOP sentinel removed"
    _log_loop_status "$lf"
    state_set step0_outcome "stopped"
    state_mark step0
    return 0
}

_check_gpu_idle() {  # PROVISIONAL: rocm-smi field layout not exercised pre-flight
    local lf="$1" use
    if [[ -n "${BOUNDARY_STUB_DIR:-}" && -x "${BOUNDARY_STUB_DIR}/rocm_smi" ]]; then
        use=$("${BOUNDARY_STUB_DIR}/rocm_smi" --showuse 2>> "$lf")
    elif command -v rocm-smi >/dev/null 2>&1; then
        use=$(rocm-smi --showuse 2>> "$lf")
    else
        log "step0: WARNING rocm-smi unavailable — skipping GPU-idle check"
        return 0
    fi
    echo "$use" >> "$lf"
    if echo "$use" | grep -qE 'GPU use.*:\s*0\s*$'; then
        log "step0: GPU use confirmed 0%"
    else
        log "step0: WARNING could not confirm GPU use 0% (see $lf) — proceeding on process-group death"
    fi
}

_log_loop_status() {
    local lf="$1"
    if [[ -f "$STORE/loop-status.json" ]]; then
        { echo "--- final loop-status dispositions ---";
          python3 -m json.tool "$STORE/loop-status.json" 2>/dev/null \
              || cat "$STORE/loop-status.json"; } >> "$lf"
        log "step0: final loop-status dispositions appended to $lf"
    else
        log "step0: no loop-status.json in $STORE"
    fi
}

# ---------------------------------------------------------------- step 1: funsafe A/B
resolve_champion_tip() { git -C "$CHAMP_TREE" rev-parse "refs/heads/$CHAMPION_BRANCH" 2>/dev/null; }

recut_admission_branch() {  # move the one-line admission commit onto the current tip
    local tip="$1" parent
    parent=$(git -C "$ADMISSION_TREE" rev-parse "$ADMISSION_REF^" 2>/dev/null)
    if [[ "$parent" == "$tip" ]]; then
        log "step1: admission branch already cut from current tip ${tip:0:12}"
        state_set step1_recut "not_needed"
        return 0
    fi
    if [[ -n "$(git -C "$ADMISSION_TREE" status --porcelain)" ]]; then
        log "step1: WARNING admission tree dirty — measuring ORIGINAL geometry (base $parent)"
        state_set step1_recut "refused_dirty_tree"
        return 0
    fi
    if [[ "$(git -C "$ADMISSION_TREE" branch --show-current)" != "$ADMISSION_REF" ]]; then
        log "step1: WARNING admission tree not on $ADMISSION_REF — measuring ORIGINAL geometry"
        state_set step1_recut "refused_wrong_branch"
        return 0
    fi
    log "step1: re-cutting $ADMISSION_REF onto current tip ${tip:0:12} (was cut from ${parent:0:12})"
    if git -C "$ADMISSION_TREE" rebase --onto "$tip" "HEAD~1" >> "$WORK_DIR/step1-funsafe.log" 2>&1; then
        state_set step1_recut "recut_onto_${tip:0:12}"
    else
        git -C "$ADMISSION_TREE" rebase --abort >> "$WORK_DIR/step1-funsafe.log" 2>&1
        log "step1: WARNING re-cut rebase failed, aborted — measuring ORIGINAL geometry"
        state_set step1_recut "rebase_failed_original_geometry"
    fi
    return 0
}

step1_funsafe() {
    if state_done step1; then log "step1: already done, skipping"; return 0; fi
    local lf="$WORK_DIR/step1-funsafe.log" tip rc verdict
    log "step1: funsafe-math admission A/B (log: $lf)"

    tip=$(resolve_champion_tip)
    if [[ -z "$tip" ]]; then
        log "step1: FAILED cannot resolve $CHAMPION_BRANCH in $CHAMP_TREE — no merge, continuing"
        state_set step1_verdict "HARNESS_NOT_RUN"; state_set step1_merged "no"
        state_mark step1; return 0
    fi
    state_set champion_tip_pre_boundary "$tip"
    recut_admission_branch "$tip"

    heavy funsafe_admission "$lf" \
        python3 "$BENCH_DIR/autokernel_funsafe_math_admission.py" \
        --admission-tree "$ADMISSION_TREE" --admission-ref "$ADMISSION_REF" \
        --model "$MODEL" --pairs 20 \
        --build-root "$FUNSAFE_BUILD_ROOT" --out "$FUNSAFE_OUT"
    rc=$?
    state_set step1_harness_rc "$rc"

    verdict=$(python3 "$HELPER" verdict "$FUNSAFE_JSON")
    state_set step1_verdict "$verdict"
    log "step1: harness rc=$rc verdict=$verdict"

    # Operator ruling, encoded EXACTLY: merge only if quality effect is REAL
    # (greedy divergence demonstrated on gfx90a) AND the harness exited cleanly.
    if [[ "$rc" -eq 0 && "$verdict" == "DIVERGENCE" ]]; then
        log "step1: divergence demonstrated + clean exit — merging flag removal onto champion"
        heavy merge_verify "$lf" \
            python3 "$HELPER" merge \
            --champ-tree "$CHAMP_TREE" --branch "$CHAMPION_BRANCH" \
            --admission-tree "$ADMISSION_TREE" --admission-ref "$ADMISSION_REF" \
            --tag "$PRE_MERGE_TAG"
        if [[ $? -eq 0 ]]; then
            state_set step1_merged "yes"
            log "step1: MERGED — champion now at $(resolve_champion_tip)"
        else
            state_set step1_merged "rolled_back"
            log "step1: merge FAILED and was rolled back to $PRE_MERGE_TAG — continuing unmerged"
        fi
    elif [[ "$rc" -ne 0 ]]; then
        state_set step1_merged "no"
        log "step1: harness FAILED (rc=$rc) — no merge, continuing with unmerged champion"
    else
        state_set step1_merged "no"
        log "step1: PARITY (quality gain undemonstrated on gfx90a) — no merge per the ruling's condition; verdict goes prominently in the morning report"
    fi
    state_mark step1
    return 0
}

# ---------------------------------------------------------------- step 2: calibrations
step2_calibrations() {
    local surface lf anchor extra merged rc
    merged=$(state_get step1_merged)
    if [[ "$merged" == "yes" ]]; then
        # champion advanced: anchor = the merged champion build (rebuilt+oracled in
        # step 1's merge). Hand-built dir, no provenance.json → needs the waiver.
        anchor="$CHAMP_TREE/build-hip"; extra="--allow-unverified-anchor"
    else
        anchor="$ANCHOR_GEN"; extra=""
    fi
    state_set step2_anchor "$anchor"

    for surface in "${CALIB_SURFACES[@]}"; do
        if state_done "step2_$surface"; then log "step2: $surface already done, skipping"; continue; fi
        lf="$WORK_DIR/step2-$surface.log"
        log "step2: calibrating $surface (anchor $anchor, log: $lf)"
        ( cd "$KERNEL_RND" && heavy calibrate_surface "$lf" \
              python3 -u -m autokernel.loop.run \
              --worktree "$CHAMP_TREE" --anchor-build "$anchor" \
              --model "$MODEL" --store "$STORE" \
              --surface "$surface" --calibrate-surface "$CALIB_PAIRS" $extra )
        rc=$?
        if [[ $rc -eq 0 && -f "$STORE/calibration/$surface.json" ]]; then
            log "step2: $surface calibrated — $STORE/calibration/$surface.json written"
            state_mark "step2_$surface"
        else
            log "step2: $surface FAILED (rc=$rc, record $([[ -f "$STORE/calibration/$surface.json" ]] && echo present || echo missing)) — continuing to next surface"
            state_set "step2_${surface}_outcome" "failed_rc$rc"
        fi
    done
    return 0
}

# ---------------------------------------------------------------- step 3: serving refresh
step3_serving_refresh() {
    if state_done step3; then log "step3: already done, skipping"; return 0; fi
    local lf="$WORK_DIR/step3-refresh.log" refresh_date pid_flag=() rc pid
    refresh_date=$(state_get refresh_date)
    if [[ -z "$refresh_date" ]]; then
        refresh_date=$(date -u +%Y%m%d)   # pinned at first attempt so a crash+relaunch
        state_set refresh_date "$refresh_date"  # past midnight keeps one artifact date
    fi
    pid=$(state_get run21_pid)
    [[ -n "$pid" ]] && pid_flag=(--loop-pid "$pid")
    # --minimal per the operator ruling (2026-08-31, boundary scope): the boundary
    # needs the headline + no-regression gate on the new champion (~2h); the full
    # 24-cell sweep is reserved for promotion time. Minimal = the concurrency grid
    # runs its kv_unified=0 half only (the half the bundle consumes), at the cost
    # of the G2 paired control; anchor validation and greedy parity run in full.
    log "step3: serving evidence refresh, MINIMAL mode, --date $refresh_date (log: $lf)"

    heavy serving_refresh "$lf" \
        python3 "$BENCH_DIR/serving_evidence_refresh.py" \
        --date "$refresh_date" --minimal "${pid_flag[@]}"
    rc=$?
    if [[ $rc -eq 0 ]]; then
        log "step3: refresh complete — bundle published"
        state_mark step3
    else
        log "step3: refresh FAILED (rc=$rc, see $lf) — continuing to the readiness report"
        state_set step3_outcome "failed_rc$rc"
    fi
    return 0
}

# ------------------------------------------------------- all-green evaluation (OP-32)
# The operator's 2026-08-31 ruling ("pre-authorize run 22 if all boundary steps
# are green", OP-32) defines an exact conjunction. This driver EVALUATES it and
# reports the verdict + every red reason; the launch itself remains with the
# operator (run starts stay operator-gated in this driver — the report carries
# the ready-to-run command). The default on ANY doubt is HOLD.
all_green_reasons() {   # one reason per line; EMPTY output == all green
    local outcome rc verdict merged surface
    outcome=$(state_get step0_outcome)
    if ! state_done step0 || [[ "$outcome" != "stopped" && "$outcome" != "already_dead" ]]; then
        echo "step0: run 21 not verified stopped (outcome=${outcome:-none})"
    fi
    rc=$(state_get step1_harness_rc); verdict=$(state_get step1_verdict)
    merged=$(state_get step1_merged)
    if [[ "$rc" != "0" ]]; then
        echo "step1: harness did not exit cleanly (rc=${rc:-never_ran}) — a crash is not green"
    elif [[ "$verdict" == "DIVERGENCE" && "$merged" != "yes" ]]; then
        echo "step1: divergence demonstrated but merge state is '$merged', not 'yes'"
    elif [[ "$verdict" != "DIVERGENCE" && "$verdict" != "PARITY" ]]; then
        echo "step1: verdict '$verdict' is neither PARITY nor DIVERGENCE"
    fi
    for surface in "${CALIB_SURFACES[@]}"; do
        if ! state_done "step2_$surface" || [[ ! -f "$STORE/calibration/$surface.json" ]]; then
            echo "step2: $surface calibration missing or failed"
        fi
    done
    if ! state_done step3; then
        echo "step3: serving bundle not sealed/published"
    fi
}

evaluate_all_green() {
    local reasons
    reasons=$(all_green_reasons)
    if [[ -z "$reasons" ]]; then
        state_set allgreen "yes"
        log "all-green: YES — every boundary step green (OP-32 condition met)"
    else
        state_set allgreen "no"
        while IFS= read -r line; do
            state_set allgreen_reason "$line"
            log "all-green: NO — $line"
        done <<< "$reasons"
    fi
    return 0
}

# ---------------------------------------------------------------- step 4: readiness report
step4_report() {
    local lf="$WORK_DIR/step4-report.log"
    log "step4: writing run-22 readiness package (no device)"
    python3 "$HELPER" report \
        --state "$STATE_FILE" --work-dir "$WORK_DIR" --store "$STORE" \
        --surface-dir "$SURFACE_DIR" --report "$REPORT" \
        --champ-tree "$CHAMP_TREE" --branch "$CHAMPION_BRANCH" \
        --model "$MODEL" --lane-root "$LANE_ROOT" --anchor-gen "$ANCHOR_GEN" \
        --funsafe-json "$FUNSAFE_JSON" >> "$lf" 2>&1
    if [[ $? -eq 0 ]]; then
        log "step4: report written to $REPORT"
        state_mark step4
    else
        log "step4: report writer FAILED (see $lf)"
    fi
    return 0
}

# ---------------------------------------------------------------- dry run
dry_run_plan() {
    local pid="(pid file $PID_FILE missing)" alive="n/a" tip parent admission
    [[ -f "$PID_FILE" ]] && pid=$(tr -dc '0-9' < "$PID_FILE")
    [[ "$pid" =~ ^[0-9]+$ ]] && { loop_alive "$pid" && alive="ALIVE (cmdline matches '$LOOP_MATCH')" || alive="dead / not the loop"; }
    tip=$(resolve_champion_tip || echo UNRESOLVED)
    admission=$(git -C "$ADMISSION_TREE" rev-parse "$ADMISSION_REF" 2>/dev/null || echo UNRESOLVED)
    parent=$(git -C "$ADMISSION_TREE" rev-parse "$ADMISSION_REF^" 2>/dev/null || echo UNRESOLVED)
    cat <<EOF
================ boundary_20260831 DRY RUN — nothing will be executed ================
work dir        $WORK_DIR   (state: $STATE_FILE)
report          $REPORT

step0  stop run 21
  pid file      $PID_FILE -> pid $pid   liveness: $alive
  T0            write $STOP_SENTINEL  AND  kill -TERM $pid
  T0+${KILL_WAIT_S}s     if alive: kill -KILL -<pgid-of-$pid>  (group; setsid'd loop => pgid==pid)
  verify        /proc/$pid gone; PGID scan for group remnants (observation only, no
                name-pattern kills); GPU use 0 via rocm-smi [PROVISIONAL: rocm-smi
                --showuse parse not exercised pre-flight]; then rm STOP sentinel;
                log final loop-status dispositions from $STORE/loop-status.json

step1  funsafe-math flag admission A/B
  champion tip  $CHAMPION_BRANCH @ ${tip:0:12} (resolved live from $CHAMP_TREE)
  admission     $ADMISSION_REF @ ${admission:0:12}, currently cut from ${parent:0:12}
  re-cut        $( [[ "$parent" == "$tip" ]] && echo "not needed (already on tip)" || echo "git -C $ADMISSION_TREE rebase --onto $tip HEAD~1   [PROVISIONAL: assumed clean — one-line change; on conflict: abort + measure original geometry]" )
  run           python3 $BENCH_DIR/autokernel_funsafe_math_admission.py \\
                    --admission-tree $ADMISSION_TREE --admission-ref $ADMISSION_REF \\
                    --model $MODEL --pairs 20 \\
                    --build-root $FUNSAFE_BUILD_ROOT \\
                    --out $FUNSAFE_OUT
  decision      (operator ruling encoded) rc==0 AND divergent_prompts>0  => MERGE:
                    python3 $HELPER merge --champ-tree $CHAMP_TREE \\
                        --branch $CHAMPION_BRANCH --admission-tree $ADMISSION_TREE \\
                        --admission-ref $ADMISSION_REF --tag $PRE_MERGE_TAG
                    (tags tip first; cherry-pick; taskset -c 96-183 cmake --build
                     build-hip -j64; claim.hold + test-backend-ops MUL_MAT oracle;
                     rollback to tag on ANY failure)
                PARITY (divergent_prompts==0)  => NO merge, verdict prominent in report
                harness rc!=0                  => NO merge, log, continue

step2  dec-b2/b4/b8 floor calibrations (D8 method, N=$CALIB_PAIRS pairs x 3 conditions each)
  anchor        merged:   $CHAMP_TREE/build-hip  + --allow-unverified-anchor
                unmerged: $ANCHOR_GEN  (provenance.json champion_commit must be
                ancestor-or-equal of tip — verified true for anchor-gen-005 @ aba5a8155cdd)
  run (each)    cd $KERNEL_RND && python3 -u -m autokernel.loop.run \\
                    --worktree $CHAMP_TREE --anchor-build <anchor> \\
                    --model $MODEL \\
                    --store $STORE \\
                    --surface <dec-b2|dec-b4|dec-b8> --calibrate-surface $CALIB_PAIRS [waiver]
  writes        $STORE/calibration/<surface>.json ; one failure logs and continues

step3  serving evidence refresh (MINIMAL mode — operator ruling 2026-08-31)
  run           python3 $BENCH_DIR/serving_evidence_refresh.py \\
                    --date <pinned at first attempt, UTC> --minimal --loop-pid $pid
                (its own preflight refuses a live loop; step0 guarantees dead)
  minimal =     concurrency grid runs kv_unified=0 only (the half the bundle
                consumes, ~1.3h instead of ~2.6h) at the cost of the G2 paired
                control; anchor validation + greedy parity still run in full.
                Full 24-cell sweep reserved for promotion time. Boundary total
                estimate ~3.5-5h.

step4  run-22 readiness package -> $REPORT
  per-step outcomes+timestamps+logs; flag verdict + merged-or-not; three floors;
  bundle headline+path; the OP-32 all-green verdict (step0 stopped+verified,
  step1 clean exit + merge-iff-divergence, all three calibration files written,
  step3 bundle published — ANY red reason is listed); exact run-22 launch
  command (tg128 primary, workers 7, pairs 20, --rank-prior-experiments).
  THIS DRIVER NEVER STARTS RUN 22 — the launch stays with the operator (the
  permission system denied encoding an unattended start; OP-32's verdict is
  evaluated and reported, the command is ready to paste).
=====================================================================================
EOF
}

# ---------------------------------------------------------------- main
main() {
    local at="" wait_s
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --at)      at="$2"; shift 2 ;;
            --now)     shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            *) echo "usage: boundary_20260831.sh [--at HH:MM] [--now] [--dry-run]" >&2; return 2 ;;
        esac
    done

    if [[ "$DRY_RUN" == 1 ]]; then dry_run_plan; return 0; fi

    mkdir -p "$WORK_DIR"
    touch "$STATE_FILE"

    if [[ -n "$at" ]]; then
        wait_s=$(seconds_until_utc "$at") || { log "bad --at time '$at'"; return 2; }
        if (( wait_s > 0 )); then
            log "waiting ${wait_s}s until ${at}Z"
            sleep "$wait_s"
        else
            log "WARNING ${at}Z already past — starting immediately"
        fi
    fi

    log "=== boundary run-21->22 driver start (state: $STATE_FILE) ==="
    if ! step0_stop_run21; then
        log "ABORT: run 21 could not be stopped; device steps skipped; writing failure report"
        evaluate_all_green
        step4_report
        state_set driver_done 1   # the launch watcher keys on this, never on greenness
        return 1
    fi
    step1_funsafe
    step2_calibrations
    step3_serving_refresh
    evaluate_all_green
    step4_report
    state_set driver_done 1   # last act; the OP-32 launch watcher waits for exactly this
    log "=== boundary driver complete — run 22 NOT started (launch stays with the operator; the report carries the OP-32 all-green verdict and the ready command) ==="
    return 0
}

if [[ "${BOUNDARY_LIB_ONLY:-0}" != 1 ]]; then
    main "$@"
    exit $?
fi
