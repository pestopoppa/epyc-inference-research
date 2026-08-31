#!/bin/bash
# test_boundary_20260831.sh — device-free test harness for boundary_20260831.sh.
#
# Runs the WHOLE driver end-to-end against a sandbox: fake git repos, fake loop
# processes (self-spawned, killed only by their self-captured pids), and stubs
# for the three heavy harnesses via BOUNDARY_STUB_DIR. No GPU, no claim, no real
# repo is touched. Also mutation-tests the merge conditional, the resume logic
# and the never-start-run-22 property.
#
#   TMPDIR=/workspace/tmp bash scripts/benchmark/test_boundary_20260831.sh
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REAL_DRIVER="$HERE/boundary_20260831.sh"
PASS=0; FAIL=0; FAILED_NAMES=()

ok()   { PASS=$((PASS+1)); }
bad()  { FAIL=$((FAIL+1)); FAILED_NAMES+=("$1"); echo "  FAIL: $1"; }
assert() {  # assert <desc> <cmd...>
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then ok; else bad "$desc"; fi
}
assert_not() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then bad "$desc"; else ok; fi
}

# ------------------------------------------------------------------ sandbox
# Unique dir per test INVOCATION (mutation re-runs included): mktemp, not a
# counter — a counter bumped inside $(...) never increments the parent shell,
# which silently reused sandboxes and made two mutants look survived.
new_T() { mktemp -d "$TB/sb-$1.XXXXXX"; }

make_sandbox() {  # make_sandbox <dir>
    local T="$1"
    mkdir -p "$T"/{work,store/calibration,surface,stubs}
    echo '{"final": "fake loop-status for tests"}' > "$T/store/loop-status.json"
    mkdir -p "$T/store/anchor-gen-005"

    # champion repo + admission worktree, champion tip advanced past the cut point
    git init -q -b master "$T/llama"
    git -C "$T/llama" config user.email t@t; git -C "$T/llama" config user.name t
    mkdir -p "$T/llama/ggml/src/ggml-hip"
    printf 'a\n-funsafe-math-optimizations\nb\n' > "$T/llama/ggml/src/ggml-hip/CMakeLists.txt"
    git -C "$T/llama" add -A; git -C "$T/llama" commit -qm base
    git -C "$T/llama" checkout -qb ak/champion/test
    git -C "$T/llama" worktree add -q -b ak/admission/test "$T/admission" HEAD
    sed -i '/funsafe-math/d' "$T/admission/ggml/src/ggml-hip/CMakeLists.txt"
    git -C "$T/admission" commit -qam "remove funsafe flag"
    echo x > "$T/llama/other.c"          # champion advances -> re-cut is needed
    git -C "$T/llama" add other.c; git -C "$T/llama" commit -qm advance

    # ---- stubs: record argv, produce the artifacts the driver checks for
    cat > "$T/stubs/funsafe_admission" <<'EOS'
#!/bin/bash
echo "funsafe_admission $*" >> "$STUB_CALLS"
out=""; prev=""
for a in "$@"; do [[ "$prev" == "--out" ]] && out="$a"; prev="$a"; done
mkdir -p "$out"
printf '{"divergent_prompts": %s, "verdict_hint": "stub"}\n' "${STUB_DIVERGE:-0}" \
    > "$out/funsafe-math-admission.json"
exit "${STUB_FUNSAFE_RC:-0}"
EOS
    cat > "$T/stubs/merge_verify" <<'EOS'
#!/bin/bash
echo "merge_verify $*" >> "$STUB_CALLS"
exit "${STUB_MERGE_RC:-0}"
EOS
    cat > "$T/stubs/calibrate_surface" <<'EOS'
#!/bin/bash
echo "calibrate_surface $*" >> "$STUB_CALLS"
surface=""; store=""; prev=""
for a in "$@"; do
    [[ "$prev" == "--surface" ]] && surface="$a"
    [[ "$prev" == "--store" ]] && store="$a"
    prev="$a"
done
mkdir -p "$store/calibration"
printf '{"floor_pct": {"5": 2.1, "20": 1.234}}\n' > "$store/calibration/$surface.json"
exit "${STUB_CALIB_RC:-0}"
EOS
    cat > "$T/stubs/serving_refresh" <<'EOS'
#!/bin/bash
echo "serving_refresh $*" >> "$STUB_CALLS"
d=""; prev=""
for a in "$@"; do [[ "$prev" == "--date" ]] && d="$a"; prev="$a"; done
printf '{"headline": {"summary": "TEST-HEADLINE +12.3%% single-stream"}}\n' \
    > "$STUB_SURFACE_DIR/operator_gate_bundle_$d.json"
exit "${STUB_REFRESH_RC:-0}"
EOS
    cat > "$T/stubs/rocm_smi" <<'EOS'
#!/bin/bash
echo "rocm_smi $*" >> "$STUB_CALLS"
echo "GPU[0] : GPU use (%): 0"
EOS
    chmod +x "$T/stubs/"*
}

start_fake_loop() {  # start_fake_loop <T> <ignore_term:0|1> — pid lands in run21.pid
    local T="$1" ignore="$2"
    rm -f "$T/run21.pid"
    if [[ "$ignore" == 1 ]]; then
        # session leader + one same-group child, both ignoring SIGTERM
        setsid bash -c "echo \$\$ > '$T/run21.pid'; trap '' TERM; sleep 300 & exec sleep 300" &
    else
        setsid bash -c "echo \$\$ > '$T/run21.pid'; exec sleep 300" &
    fi
    for _ in $(seq 50); do [[ -s "$T/run21.pid" ]] && break; sleep 0.1; done
}

run_driver() {  # run_driver <T> <driver> [extra driver args...] — env from sandbox
    local T="$1" driver="$2"; shift 2
    ( export BOUNDARY_LANE_ROOT="${BOUNDARY_TEST_LANE_ROOT:-$(cd "$HERE/../.." && pwd)}" \
             BOUNDARY_WORK_DIR="$T/work" BOUNDARY_PID_FILE="$T/run21.pid" \
             BOUNDARY_STORE="$T/store" BOUNDARY_SURFACE_DIR="$T/surface" \
             BOUNDARY_REPORT="$T/report.md" BOUNDARY_CHAMP_TREE="$T/llama" \
             BOUNDARY_CHAMPION_BRANCH="ak/champion/test" \
             BOUNDARY_ADMISSION_TREE="$T/admission" \
             BOUNDARY_ADMISSION_REF="ak/admission/test" \
             BOUNDARY_ANCHOR_GEN="$T/store/anchor-gen-005" \
             BOUNDARY_STUB_DIR="$T/stubs" STUB_CALLS="$T/calls.log" \
             STUB_SURFACE_DIR="$T/surface" \
             BOUNDARY_KILL_WAIT_S=2 BOUNDARY_KILL_GRACE_S=3 BOUNDARY_POLL_S=1 \
             BOUNDARY_LOOP_MATCH="sleep 300"
      bash "$driver" "$@" )
}

# The property that must hold after EVERY driver run: run 22 was never launched.
assert_run22_never_launched() {  # <name> <T>
    local name="$1" T="$2"
    if [[ -f "$T/calls.log" ]] && grep "autokernel.loop.run" "$T/calls.log" \
            | grep -qv -- "--calibrate-surface"; then
        bad "$name: loop.run invoked WITHOUT --calibrate-surface (run-22-shaped launch)"
    else ok; fi
    assert_not "$name: no argv mentions run22" grep -q "run22" "$T/calls.log"
    if [[ -f "$T/report.md" ]]; then
        assert "$name: report carries the operator-gate PENDING line" \
            grep -q "awaits the operator's explicit go" "$T/report.md"
    fi
}

# ------------------------------------------------------------------ tests
t_time_wait() {
    local target got
    target=$(date -u -d "today 23:59" +%s)
    got=$(BOUNDARY_LIB_ONLY=1 source "$REAL_DRIVER"; BOUNDARY_FAKE_NOW=$((target-73)) seconds_until_utc 23:59)
    assert "time-wait: 73s before target" [ "$got" = "73" ]
    got=$(BOUNDARY_LIB_ONLY=1 source "$REAL_DRIVER"; BOUNDARY_FAKE_NOW=$((target+10)) seconds_until_utc 23:59)
    assert "time-wait: past target clamps to 0" [ "$got" = "0" ]
}

t_ordering_divergence_merge() {
    local T; T=$(new_T ordering); make_sandbox "$T"; start_fake_loop "$T" 0
    local pid; pid=$(cat "$T/run21.pid")
    STUB_DIVERGE=3 run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    local rc=$?
    assert "ordering: driver exits 0" [ "$rc" = 0 ]
    assert "ordering: fake loop dead" bash -c "! kill -0 $pid 2>/dev/null"
    assert "ordering: SIGTERM sufficed (no escalation)" \
        grep -q "step0_escalated=none_sigterm_sufficed" "$T/work/state"
    assert "ordering: STOP sentinel removed" bash -c "[[ ! -e '$T/store/STOP' ]]"
    local seq
    seq=$(awk '{print $1}' "$T/calls.log" | tr '\n' ' ')
    assert "ordering: step sequence rocm->funsafe->merge->calib x3->refresh" \
        [ "$seq" = "rocm_smi funsafe_admission merge_verify calibrate_surface calibrate_surface calibrate_surface serving_refresh " ]
    local surfaces
    surfaces=$(grep calibrate_surface "$T/calls.log" | grep -o 'dec-b[0-9]*' | tr '\n' ' ')
    assert "ordering: calibrations dec-b2 dec-b4 dec-b8 in order" \
        [ "$surfaces" = "dec-b2 dec-b4 dec-b8 " ]
    assert "divergence: merge_verify called exactly once" \
        [ "$(grep -c '^merge_verify' "$T/calls.log")" = 1 ]
    assert "divergence: merged-case calibrations use champ build-hip + waiver" \
        bash -c "! grep '^calibrate_surface' '$T/calls.log' | grep -v 'build-hip' | grep -q . \
                 && ! grep '^calibrate_surface' '$T/calls.log' | grep -v -- '--allow-unverified-anchor' | grep -q ."
    # re-cut: admission branch parent must now be the champion tip
    assert "re-cut: admission branch rebased onto current tip" \
        [ "$(git -C "$T/admission" rev-parse ak/admission/test^)" = "$(git -C "$T/llama" rev-parse ak/champion/test)" ]
    assert "report: exists with merged=yes" grep -q "Merged onto champion: \*\*yes\*\*" "$T/report.md"
    assert "report: bundle headline surfaced" grep -q "TEST-HEADLINE" "$T/report.md"
    assert "report: calibration floor rows present" grep -q "20p: 1.234%" "$T/report.md"
    assert "step0: loop-status dispositions logged" \
        grep -q "fake loop-status for tests" "$T/work/step0-stop.log"
    assert "ordering: OP-32 verdict ALL GREEN in report" grep -q "ALL GREEN" "$T/report.md"
    assert "ordering: state allgreen=yes" grep -q "^allgreen=yes$" "$T/work/state"
    # operator ruling 2026-08-31: the boundary refresh runs --minimal, and the
    # report must say so plainly (measured vs skipped) so nobody mistakes the
    # bundle for the full evidence sweep.
    assert "refresh: invoked with --minimal" \
        bash -c "grep '^serving_refresh' '$T/calls.log' | grep -q -- '--minimal'"
    assert "report: names the MINIMAL refresh and the skipped G2 paired control" \
        bash -c "grep -q 'MINIMAL refresh' '$T/report.md' && grep -q 'kv_unified=1 paired control' '$T/report.md'"
    assert_run22_never_launched "ordering" "$T"
}

t_kill_escalation() {
    local T; T=$(new_T escalate); make_sandbox "$T"; start_fake_loop "$T" 1
    local pid; pid=$(cat "$T/run21.pid")
    STUB_DIVERGE=0 run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    assert "escalation: leader dead after SIGKILL group" bash -c "! kill -0 $pid 2>/dev/null"
    assert "escalation: state records group SIGKILL" \
        grep -q "step0_escalated=sigkill_group_$pid" "$T/work/state"
    local survivors
    survivors=$(ps -eo pid=,pgid= | awk -v g="$pid" '$2==g {print $1}')
    assert "escalation: no group survivors (children killed too)" [ -z "$survivors" ]
    assert_run22_never_launched "escalation" "$T"
}

t_parity_no_merge() {
    local T; T=$(new_T parity); make_sandbox "$T"; start_fake_loop "$T" 0
    STUB_DIVERGE=0 run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    assert_not "parity: merge_verify NOT called" grep -q '^merge_verify' "$T/calls.log"
    assert "parity: verdict recorded" grep -q "step1_verdict=PARITY" "$T/work/state"
    assert "parity: report carries the PROMINENT parity verdict" \
        grep -q "PROMINENT — FLAG VERDICT: PARITY" "$T/report.md"
    assert "parity: calibrations anchored on anchor-gen-005" \
        bash -c "! grep '^calibrate_surface' '$T/calls.log' | grep -v 'anchor-gen-005' | grep -q ."
    assert_not "parity: no --allow-unverified-anchor without a merge" \
        grep -q -- '--allow-unverified-anchor' "$T/calls.log"
    assert "parity: OP-32 verdict ALL GREEN (either verdict counts as green)" \
        grep -q "ALL GREEN" "$T/report.md"
    assert_run22_never_launched "parity" "$T"
}

t_harness_fail_no_merge() {
    local T; T=$(new_T hfail); make_sandbox "$T"; start_fake_loop "$T" 0
    # JSON claims divergence but the harness exits dirty: the ruling requires a
    # clean exit AND divergence — no merge.
    STUB_DIVERGE=5 STUB_FUNSAFE_RC=3 run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    assert_not "harness-fail: merge_verify NOT called despite divergence in JSON" \
        grep -q '^merge_verify' "$T/calls.log"
    assert "harness-fail: rc recorded" grep -q "step1_harness_rc=3" "$T/work/state"
    assert "harness-fail: later steps still ran" grep -q '^serving_refresh' "$T/calls.log"
    assert "harness-fail: OP-32 HOLD with a step1 reason" \
        bash -c "grep -q 'NOT all green' '$T/report.md' && grep -q 'step1: harness did not exit cleanly' '$T/report.md'"
    assert_run22_never_launched "harness-fail" "$T"
}

t_calib_failure_continues() {
    local T; T=$(new_T calfail); make_sandbox "$T"; start_fake_loop "$T" 0
    STUB_DIVERGE=0 STUB_CALIB_RC=1 run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    assert "calib-fail: all three surfaces attempted" \
        [ "$(grep -c '^calibrate_surface' "$T/calls.log")" = 3 ]
    assert "calib-fail: refresh still ran" grep -q '^serving_refresh' "$T/calls.log"
    assert "calib-fail: report marks surfaces failed" \
        grep -q "step2_dec-b2_outcome=failed_rc1" "$T/work/state"
    assert "calib-fail: OP-32 HOLD with step2 reasons" \
        bash -c "grep -q 'NOT all green' '$T/report.md' && grep -q 'step2: dec-b2 calibration missing or failed' '$T/report.md'"
    assert_run22_never_launched "calib-fail" "$T"
}

t_refresh_fail_holds() {
    local T; T=$(new_T reffail); make_sandbox "$T"; start_fake_loop "$T" 0
    STUB_DIVERGE=0 STUB_REFRESH_RC=1 run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    assert "refresh-fail: OP-32 HOLD with step3 reason" \
        bash -c "grep -q 'NOT all green' '$T/report.md' && grep -q 'step3: serving bundle not sealed/published' '$T/report.md'"
    assert_not "refresh-fail: report does not claim ALL GREEN" grep -q "ALL GREEN" "$T/report.md"
    assert_run22_never_launched "refresh-fail" "$T"
}

t_merge_rollback_holds() {
    local T; T=$(new_T rollback); make_sandbox "$T"; start_fake_loop "$T" 0
    STUB_DIVERGE=3 STUB_MERGE_RC=1 run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    assert "rollback: merge attempted" grep -q '^merge_verify' "$T/calls.log"
    assert "rollback: state merged=rolled_back" grep -q "step1_merged=rolled_back" "$T/work/state"
    assert "rollback: OP-32 HOLD — divergence without a landed merge is not green" \
        bash -c "grep -q 'NOT all green' '$T/report.md' && grep -q 'divergence demonstrated but merge state' '$T/report.md'"
    assert_run22_never_launched "rollback" "$T"
}

t_idempotent_rerun() {
    local T; T=$(new_T idem); make_sandbox "$T"; start_fake_loop "$T" 0
    STUB_DIVERGE=0 run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    : > "$T/calls.log"
    run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    local rc=$?
    assert "idempotent: rerun exits 0" [ "$rc" = 0 ]
    assert "idempotent: rerun makes ZERO heavy calls" [ ! -s "$T/calls.log" ]
}

t_crash_resume() {
    local T; T=$(new_T resume); make_sandbox "$T"
    rm -f "$T/run21.pid"                       # run 21 long dead on relaunch
    mkdir -p "$T/work"
    printf 'step0=done\nrun21_pid=99999999\nstep1_verdict=PARITY\nstep1_merged=no\nstep1=done\n' > "$T/work/state"
    run_driver "$T" "$REAL_DRIVER" --now >/dev/null 2>&1
    assert_not "resume: funsafe NOT re-run" grep -q '^funsafe_admission' "$T/calls.log"
    assert "resume: calibrations ran" [ "$(grep -c '^calibrate_surface' "$T/calls.log")" = 3 ]
    assert "resume: refresh ran" grep -q '^serving_refresh' "$T/calls.log"
    assert "resume: incomplete seeded state is HOLD not green (doubt defaults to HOLD)" \
        grep -q "NOT all green" "$T/report.md"
    assert_run22_never_launched "resume" "$T"
}

t_dry_run() {
    local T; T=$(new_T dry); make_sandbox "$T"
    echo 4242 > "$T/run21.pid"
    local out rc
    out=$(run_driver "$T" "$REAL_DRIVER" --dry-run 2>&1); rc=$?
    assert "dry-run: exits 0" [ "$rc" = 0 ]
    assert "dry-run: prints the plan" bash -c "echo \"\$1\" | grep -q 'DRY RUN'" _ "$out"
    assert "dry-run: resolves the pid" bash -c "echo \"\$1\" | grep -q 'pid 4242'" _ "$out"
    assert "dry-run: names PROVISIONAL items" bash -c "echo \"\$1\" | grep -q 'PROVISIONAL'" _ "$out"
    assert "dry-run: no work dir created" bash -c "[[ ! -e '$T/work/state' ]]"
    assert "dry-run: no heavy calls" bash -c "[[ ! -f '$T/calls.log' ]]"
}

t_verdict_helper() {
    local T; T=$(new_T verdict); mkdir -p "$T"
    echo '{"divergent_prompts": 2}' > "$T/d.json"
    echo '{"divergent_prompts": 0}' > "$T/p.json"
    echo 'garbage' > "$T/u.json"
    assert "verdict: divergence" \
        [ "$(python3 "$HERE/boundary_20260831_helper.py" verdict "$T/d.json")" = "DIVERGENCE" ]
    assert "verdict: parity" \
        [ "$(python3 "$HERE/boundary_20260831_helper.py" verdict "$T/p.json")" = "PARITY" ]
    assert "verdict: unreadable" \
        [ "$(python3 "$HERE/boundary_20260831_helper.py" verdict "$T/u.json")" = "UNREADABLE" ]
    assert "verdict: missing file unreadable" \
        [ "$(python3 "$HERE/boundary_20260831_helper.py" verdict "$T/nope.json")" = "UNREADABLE" ]
}

# ------------------------------------------------------------------ mutations
MUT_KILLED=0; MUT_SURVIVED=0
mutate_and_expect_kill() {  # <name> <sed-script> <test-fn>
    local name="$1" sedscript="$2" testfn="$3"
    local mutant="$TB/mutant-$name.sh"
    sed "$sedscript" "$REAL_DRIVER" > "$mutant"
    if cmp -s "$mutant" "$REAL_DRIVER"; then
        bad "mutation $name: sed did not change the driver (stale pattern)"; return
    fi
    bash -n "$mutant" || { bad "mutation $name: mutant does not parse"; return; }
    local before_fail=$FAIL before_pass=$PASS before_names=${#FAILED_NAMES[@]}
    REAL_DRIVER="$mutant" "$testfn" >/dev/null
    if (( FAIL > before_fail )); then
        # the suite caught the mutant: absorb its expected failures into a KILL
        FAILED_NAMES=("${FAILED_NAMES[@]:0:before_names}")
        FAIL=$before_fail; PASS=$before_pass
        MUT_KILLED=$((MUT_KILLED+1))
        echo "  mutation $name: KILLED"
    else
        PASS=$before_pass
        MUT_SURVIVED=$((MUT_SURVIVED+1)); bad "mutation $name: SURVIVED (tests blind to it)"
    fi
}

run_mutations() {
    # The merge-conditional line is targeted whole, so the all-green copy of the
    # same comparison is left intact — each mutant changes exactly one decision.
    # M1: merge fires regardless of verdict -> parity test must catch
    mutate_and_expect_kill always_merge \
        's/if \[\[ "\$rc" -eq 0 && "\$verdict" == "DIVERGENCE" \]\]; then/if [[ "$rc" -eq 0 ]]; then/' \
        t_parity_no_merge
    # M2: conditional inverted (merge on parity only) -> divergence test must catch
    mutate_and_expect_kill inverted_merge \
        's/if \[\[ "\$rc" -eq 0 && "\$verdict" == "DIVERGENCE" \]\]; then/if [[ "$rc" -eq 0 \&\& "$verdict" == "PARITY" ]]; then/' \
        t_ordering_divergence_merge
    # M3: rc guard dropped (merge despite harness failure) -> harness-fail test must catch
    mutate_and_expect_kill ignore_rc \
        's/if \[\[ "\$rc" -eq 0 && "\$verdict" == "DIVERGENCE" \]\]; then/if [[ "$verdict" == "DIVERGENCE" ]]; then/' \
        t_harness_fail_no_merge
    # M4: a run-22 launch smuggled in after the refresh -> never-launch property must catch
    mutate_and_expect_kill launches_run22 \
        's|^    step3_serving_refresh$|    step3_serving_refresh\n    ( cd "$KERNEL_RND" \&\& heavy calibrate_surface "$WORK_DIR/run22.log" python3 -u -m autokernel.loop.run --store "$STORE" --iterations 0 --out "$STORE/run22" )|' \
        t_ordering_divergence_merge
    # M5: step1 resume guard broken -> crash-resume test must catch
    mutate_and_expect_kill resume_guard \
        's/if state_done step1; then/if false; then/' t_crash_resume
    # M6: all-green ignores a failed serving refresh -> refresh-fail test must catch
    mutate_and_expect_kill green_ignores_step3 \
        's/if ! state_done step3; then/if false; then/' t_refresh_fail_holds
    # M7: all-green forced yes -> any red case must catch (harness crash here)
    mutate_and_expect_kill green_forced_yes \
        's/if \[\[ -z "\$reasons" \]\]; then/if true; then/' t_harness_fail_no_merge
    # M8: all-green ignores missing calibrations -> calib-fail test must catch
    mutate_and_expect_kill green_ignores_calib \
        's/echo "step2: \$surface calibration missing or failed"/:/' t_calib_failure_continues
    # M9: --minimal dropped from the refresh (silent full-grid regression) ->
    # the ordering test's argv assertion must catch
    mutate_and_expect_kill refresh_not_minimal \
        's/--date "\$refresh_date" --minimal/--date "$refresh_date"/' t_ordering_divergence_merge
}

# ------------------------------------------------------------------ main
TB=$(mktemp -d "${TMPDIR:-/tmp}/boundary-test.XXXXXX")
trap 'rm -rf "$TB"' EXIT
HELPER_PY="$HERE/boundary_20260831_helper.py"

echo "== unit + end-to-end (sandbox: $TB) =="
t_time_wait
t_verdict_helper
t_dry_run
t_ordering_divergence_merge
t_kill_escalation
t_parity_no_merge
t_harness_fail_no_merge
t_calib_failure_continues
t_refresh_fail_holds
t_merge_rollback_holds
t_idempotent_rerun
t_crash_resume

echo "== mutation tests =="
run_mutations

echo
echo "PASS=$PASS FAIL=$FAIL  mutations: killed=$MUT_KILLED survived=$MUT_SURVIVED"
if (( FAIL > 0 || MUT_SURVIVED > 0 )); then
    printf 'failed: %s\n' "${FAILED_NAMES[@]}"
    exit 1
fi
exit 0
