#!/bin/bash
# boundary_20260831_launch_watcher.sh — the OP-32 pre-authorized run-22 launch.
#
# SEPARATE from the boundary driver on purpose. The driver (81 assertions, 9/9
# mutants) evaluates the all-green conjunction and REPORTS it; encoding the
# launch inside it was refused by the subagent that wrote it, correctly, because
# an inter-agent message is not operator consent. This watcher carries the
# operator's DIRECT ruling instead, received by the owning session 2026-08-31:
#
#   "pre-authorize run 22 if all boundary steps are green"   (OP-32, resolved)
#
# It waits for the driver to finish, reads the driver's own verdict, and:
#   allgreen == yes  -> dry-run refusal check against the final champion state;
#                       exit 0 with "— verified" REQUIRED; then launch run 22.
#   anything else    -> touch HOLD marker, exit 0. The morning report explains.
#
# The launch is the ONLY thing this script adds. It never re-derives greenness —
# the driver's state file is the single authority, so the mutation-tested
# conjunction cannot be second-guessed here.
set -uo pipefail

WORK="${BOUNDARY_WORK:-/mnt/raid0/llm/tmp/boundary-20260831}"   # override = test seam
STATE="$WORK/state"
LANE=/mnt/raid0/llm/worktrees/mains/ak-rebuild-research
KRND="${BOUNDARY_KRND:-$LANE/scripts/kernel_rnd}"
MODEL=/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
STORE="${BOUNDARY_STORE:-/mnt/raid0/llm/autokernel/loop-memory}"
RUNLOG="${BOUNDARY_RUNLOG:-/mnt/raid0/llm/tmp/run22.log}"
PIDFILE="${BOUNDARY_PIDFILE:-/mnt/raid0/llm/tmp/run22.pid}"
LOG="$WORK/launch-watcher.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# EXACTLY the driver's own accessor (key=value lines, last write wins). The first
# draft parsed whitespace fields and would have matched nothing — the
# fixture-invents-the-spelling defect, caught by reading the producer.
state_get() { [ -f "$STATE" ] && grep "^$1=" "$STATE" | tail -1 | cut -d= -f2- || true; }
DEADLINE_EPOCH="${BOUNDARY_DEADLINE_EPOCH:-$(date -u -d "2026-09-01 08:00" +%s)}"

# ------------------------------------------------------------- wait for the driver
# The driver writes `driver_done 1` as its last act (verified in its source).
mkdir -p "$WORK"
say "watcher armed; waiting for driver_done"
while [ "$(state_get driver_done)" != "1" ]; do
    if [ "$(date -u +%s)" -ge "$DEADLINE_EPOCH" ]; then
        say "HOLD: driver did not finish by 08:00Z deadline — run 22 NOT started"
        touch "$WORK/RUN22-HELD"; exit 0
    fi
    sleep 60
done
say "driver finished; allgreen=$(state_get allgreen)"

if [ "$(state_get allgreen)" != "yes" ]; then
    say "HOLD: all-green not met — run 22 NOT started (reasons in the report)"
    touch "$WORK/RUN22-HELD"
    exit 0
fi

# ----------------------------------------------------- resolve the final anchor
# BOTH branches use champ2/build-hip at the tip, rebuilt incrementally here.
# Rewritten 21:40Z on 2026-08-31, minutes before arming: the original unmerged
# path pinned anchor-gen-005, which the run-21 loop PRUNED when gens 006/007
# were promoted — and gen-007 was then guard-REFUSED (+1.765% between two builds
# of one commit, run 21 aborted itself). Anchoring run 22 on the surviving
# gen-006 (32fad018) while the branch tip is 14ba0262 would re-create the
# stale-anchor defect in miniature: every candidate would carry the fourth
# keep's marginal baked in. A fresh tip build with the unverified-anchor waiver
# is honest instead: run 22's startup refusal still checks provenance ancestry,
# and its FIRST promotion guard re-verifies the anchor by measurement — the
# same guard that just proved it catches a bad anchor in one iteration.
ANCHOR=/mnt/raid0/llm/tmp/champ2/build-hip
WAIVER=(--allow-unverified-anchor)
say "rebuilding champ2/build-hip at the branch tip (incremental; no-op if step 3 left it current)"
taskset -c 96-183 cmake --build /mnt/raid0/llm/tmp/champ2/build-hip -j64 >> "$LOG" 2>&1
BRC=$?
if [ $BRC -ne 0 ]; then
    say "HOLD: tip rebuild failed (rc=$BRC) — run 22 NOT started"
    touch "$WORK/RUN22-HELD"; exit 0
fi
say "anchor: $ANCHOR ${WAIVER[*]:-} (tip build, first-keep guard re-verifies)"

# ------------------------------------------------- final refusal check (dry-run)
cd "$KRND" || { say "HOLD: cannot cd $KRND"; touch "$WORK/RUN22-HELD"; exit 0; }
DRY=$(python3 -m autokernel.loop.run --worktree /mnt/raid0/llm/tmp/champ2 \
        --anchor-build "$ANCHOR" --model "$MODEL" --store "$STORE" \
        --iterations 0 --surface tg128 --pairs 20 "${WAIVER[@]}" --dry-run 2>&1)
RC=$?
echo "$DRY" >> "$LOG"
if [ $RC -ne 0 ] || ! grep -q "— verified" <<< "$DRY"; then
    say "HOLD: final dry-run refusal check failed (rc=$RC) — run 22 NOT started"
    touch "$WORK/RUN22-HELD"
    exit 0
fi
say "dry-run verified — launching run 22 under OP-32"

# ------------------------------------------------------------------- the launch
setsid nohup python3 -u -m autokernel.loop.run \
  --worktree /mnt/raid0/llm/tmp/champ2 \
  --anchor-build "$ANCHOR" "${WAIVER[@]}" \
  --model "$MODEL" --store "$STORE" \
  --iterations 0 --surface tg128 --pairs 20 --workers 7 --rank-prior-experiments \
  --worker-root /mnt/raid0/llm/tmp/ak-lanes \
  --worker-build-root /mnt/raid0/llm/tmp/ak-lane-builds \
  --out "$STORE/run22" > "$RUNLOG" 2>&1 < /dev/null &

sleep 20
PID=""
for p in /proc/[0-9]*; do
    [ -r "$p/cmdline" ] || continue   # procs vanish mid-scan; a failed redirect is noise, not signal
    cl=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$cl" in *"autokernel.loop.run"*run22*) PID=${p#/proc/} ;; esac
done
if [ -n "$PID" ]; then
    echo "$PID" > "$PIDFILE"
    say "run 22 LAUNCHED pid=$PID (OP-32 all-green)"
    head -8 "$RUNLOG" >> "$LOG"
else
    say "LAUNCH FAILED: no run22 process found after 20s — see $RUNLOG"
    touch "$WORK/RUN22-LAUNCH-FAILED"
fi
