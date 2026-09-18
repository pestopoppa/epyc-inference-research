#!/bin/bash
# INF-70 / MEAS-1: build under the bench region lock.
#
# WHY: a `cmake --build -j 40` puts cc1plus-weight load on the physical cores a
# bench arm is timing. Measured 2026-09-07 (HARNESS-1): an arm holding all four
# regions still saw a peak of 3218% foreign CPU -- 32 cores, a third of the
# bench region -- from an unlocked compile plus another agent's test-backend-ops.
# A/A spread across three arms was 16.47%; excluding the compiler-contended arm
# it was 1.94%. The discriminator is the PEAK, not the median: unpinned noise
# (ps, htop) dominates the median and moves nothing.
#
# The `loadavg < 10` pre-load gate does NOT protect against this -- the gate
# passes, then the burst arrives DURING the measurement. Only in-window sampling
# catches it.
#
# There is nowhere to fence to: every logical CPU in 96-191 is the SMT sibling of
# one in 0-95 (96<->0 ... 191<->95, verified against thread_siblings_list). So a
# build cannot be moved off the bench cores -- it can only be SERIALIZED against
# them, which is what this wrapper does.
#
# Audit 2026-09-07: 19 of 21 INF-70 build scripts took no lock at all.
#
# Usage:  build_locked.sh <build-dir> [cmake --build args...]
#   e.g.  build_locked.sh "$W/build-cpu" -j 40 --target llama-server llama-bench
#
# REPO PATH: epyc-inference-research/scripts/lib/build_locked.sh
# Landed 2026-09-14 by WRAP-10 / MEAS-2 (drafted at /mnt/raid0/llm/tmp/inf70/build_locked.sh).
# ADOPTED as the campaign's standing build idiom: every kernel/bench build goes through
# this wrapper. The 19 unlocked one-shot scratch build scripts are deliberately NOT
# retrofitted -- they are spent; the convention is what had to outlive them.
#
# `region-lock` lives in the SIBLING repo epyc-orchestrator, so its path is RESOLVED
# from candidates below instead of hardcoded to one clone root (a scratch-relative or
# single-root path is what made this wrapper unusable from a worktree). Resolution is
# FAIL-CLOSED: no lock tool found means no build, never an unlocked build.
#
# Verified 2026-09-07 to actually EXCLUDE rather than merely appear to: the lock
# module's occupancy registry computes blockers from REGION OVERLAP regardless of role
# (`cpu_region_lock.py`, the `overlaps` set); role is attribution only, and only a
# `shared=True` same-role request can join a cohort. So role=build genuinely queues
# behind role=bench, even though the CLI help calls it a "per-role lock" and the lock
# FILES are per-role (`cpu_region.{role}.{region}.lock`).
set -euo pipefail

# Explicit override first, then the two canonical clone roots, then the sibling of this
# repo's own root (so a worktree beside the orchestrator clone still resolves).
_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
_siblings="$(dirname "$_repo_root")"
RL=""
for _cand in \
    ${REGION_LOCK:-} \
    /workspace/repos/epyc-orchestrator/scripts/region-lock \
    /mnt/raid0/llm/epyc-orchestrator/scripts/region-lock \
    "$_siblings/epyc-orchestrator/scripts/region-lock"; do
  if [ -n "$_cand" ] && [ -x "$_cand" ]; then RL="$_cand"; break; fi
done

AGENT="${AGENT_ID:-${INF70_AGENT:-inf70-build}}"
BUILD_DIR="${1:?usage: build_locked.sh <build-dir> [cmake --build args...]}"
shift

if [ -z "$RL" ]; then
  {
    echo "build_locked: region-lock not found -- refusing to build unlocked."
    echo "build_locked: tried \$REGION_LOCK, /workspace/repos/epyc-orchestrator,"
    echo "build_locked:       /mnt/raid0/llm/epyc-orchestrator, $_siblings/epyc-orchestrator"
    echo "build_locked: an unlocked build corrupts every concurrent sub-5% measurement."
  } >&2
  exit 2
fi

# role=build, not bench: builds queue against bench arms without claiming to be one.
# --timeout-s 0 blocks in the FAIR QUEUE. Never poll: a poller can starve forever,
# because a holder can release and immediately re-take the region.
exec "$RL" run --cpu-list 0-95 --role build --timeout-s 0 \
     --tag "build:${AGENT}" -- \
     cmake --build "$BUILD_DIR" "$@"
