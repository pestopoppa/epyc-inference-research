#!/bin/bash
# run_v7_runner_repin_20260916.sh — the operator's single command for the v7 runner re-seal
# (operator decision 2026-09-16: RE-SEAL pins 6dea92dd and 79721927 to 20a97fbd).
#
#   bash /mnt/raid0/llm/epyc-inference-research/scripts/operator/run_v7_runner_repin_20260916.sh --operator <your-name>
#
# It runs from any directory, as long as /mnt/raid0/llm/epyc-inference-research and
# /mnt/raid0/llm/epyc-root are available. Steps:
#   1. git fetch origin in /mnt/raid0/llm/epyc-inference-research
#   2. create worktree /mnt/raid0/llm/worktrees/op-v7-runner-repin-20260916 on a NEW branch
#      op/v7-runner-repin-apply from origin/main. If the worktree already exists, it refuses
#      unless --resume is given.
#   3. show the ratify script's dry-run: preflight, equivalence evidence and the full re-pin diff
#   4. ask you to type RESEAL, read from your terminal
#   5. run the ratify script with --apply and RATIFY_OPERATOR=<your-name>. This re-runs the
#      validations, zero inference, in about 3 minutes, and writes both receipts.
#   6. stage EXACTLY the six re-seal paths, show the stat, and refuse unless it is those 6 files
#   7. commit with the RATIFIED message and an "Operator-applied by <your-name>" trailer
#   8. STOP, printing the branch and commit SHA. A session pushes and merges it.
#
# It never pushes. It never touches the shared research working tree: every write lands in the
# new worktree, and the only other effects are refs and objects in the shared .git (fetch, a new
# worktree, a new branch). The re-pin itself is made by
# scripts/operator/ratify_v7_runner_repin_20260916.sh, which pins every hash and rolls back on
# any failure. epyc-root is only read, for the section-5 receipt tool and MEASUREMENT.md.
#
# Options:
#   --operator <name>  required; recorded in the receipts and the commit trailer
#   --resume           reuse an existing op-v7-runner-repin worktree (after an interrupted run)
#
# Test hooks (NOT for the operator): only when V7_REPIN_WRAPPER_TEST=1 are these honored:
#   --yes, which skips the tty prompt, and the env overrides RESEARCH_OVERRIDE, WT_DIR_OVERRIDE,
#   OP_BRANCH_OVERRIDE, EPYC_ROOT_OVERRIDE and LIVE_OVERRIDE.
# Without that variable, --yes is refused and the paths are fixed.
set -euo pipefail

TEST_MODE="${V7_REPIN_WRAPPER_TEST:-0}"
RESEARCH="/mnt/raid0/llm/epyc-inference-research"
WT_DIR="/mnt/raid0/llm/worktrees/op-v7-runner-repin-20260916"
OP_BRANCH="op/v7-runner-repin-apply"
EPYC_ROOT="/mnt/raid0/llm/epyc-root"
LIVE="/mnt/raid0/llm/epyc-inference-research"
if [ "$TEST_MODE" = "1" ]; then
  RESEARCH="${RESEARCH_OVERRIDE:-$RESEARCH}"
  WT_DIR="${WT_DIR_OVERRIDE:-$WT_DIR}"
  OP_BRANCH="${OP_BRANCH_OVERRIDE:-$OP_BRANCH}"
  EPYC_ROOT="${EPYC_ROOT_OVERRIDE:-$EPYC_ROOT}"
  LIVE="${LIVE_OVERRIDE:-$LIVE}"
fi
RATIFY_REL="scripts/operator/ratify_v7_runner_repin_20260916.sh"
COMMIT_PATHS=(
  "scripts/benchmark/dflash2_followups.py"
  "scripts/benchmark/test_dflash2_followups.py"
  "artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json"
  "artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json.sha256"
  "artifacts/operator/ratify_v7_runner_repin_20260916.json"
  "artifacts/operator/ratify_v7_runner_repin_20260916.receipt.json"
)
SUBJECT="RATIFIED: re-seal v7 runner pins 6dea92dd/79721927 -> 20a97fbd (2026-09-16)"

die() { printf '\nREFUSING: %b\n' "$*" >&2; exit 65; }
say() { printf '%s\n' "$*"; }
hdr() { printf '\n==== %s ====\n' "$*"; }

OPERATOR=""; RESUME=0; YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --operator) [ $# -ge 2 ] || die "--operator needs a name"; OPERATOR="$2"; shift 2 ;;
    --resume)   RESUME=1; shift ;;
    --yes)      [ "$TEST_MODE" = "1" ] || die "--yes is a test-only flag; the operator path is interactive"
                YES=1; shift ;;
    -h|--help)  sed -n '2,36p' "$0"; exit 0 ;;
    *) die "unknown argument: $1 (usage: $0 --operator <name> [--resume])" ;;
  esac
done
[ -n "$OPERATOR" ] || die "--operator <name> is required (usage: $0 --operator <name> [--resume])"
case "$OPERATOR" in *$'\n'*|*[[:cntrl:]]*) die "operator name contains control characters" ;; esac

[ -d "$RESEARCH/.git" ] || [ -f "$RESEARCH/.git" ] || die "$RESEARCH is not a git checkout"
[ -f "$EPYC_ROOT/scripts/operator/ratification_receipt.py" ] || die "$EPYC_ROOT has no scripts/operator/ratification_receipt.py"
G() { git -C "$WT_DIR" "$@"; }

# ------------------------------------------------------------------ 1. fetch
hdr "1/7 fetch origin ($RESEARCH)"
git -C "$RESEARCH" fetch origin || die "git fetch origin failed; nothing created"
ORIGIN_MAIN="$(git -C "$RESEARCH" rev-parse origin/main)"
say "origin/main = $ORIGIN_MAIN"

# ------------------------------------------------------------------ 2. worktree
hdr "2/7 worktree $WT_DIR (branch $OP_BRANCH)"
if [ -e "$WT_DIR" ]; then
  [ "$RESUME" -eq 1 ] || die "$WT_DIR already exists. If a previous run was interrupted, re-run with --resume; otherwise inspect it first."
  cur="$(G rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  [ "$cur" = "$OP_BRANCH" ] || die "--resume: $WT_DIR is on '$cur', not $OP_BRANCH"
  say "resuming the existing worktree on $OP_BRANCH ($(G rev-parse --short HEAD))"
else
  [ "$RESUME" -eq 0 ] || die "--resume given, but $WT_DIR does not exist"
  if git -C "$RESEARCH" show-ref --verify --quiet "refs/heads/$OP_BRANCH"; then
    die "branch $OP_BRANCH already exists without the worktree. Inspect it (git -C $RESEARCH log -3 $OP_BRANCH) and delete it by hand if it is stale."
  fi
  mkdir -p "$(dirname "$WT_DIR")"
  git -C "$RESEARCH" worktree add -b "$OP_BRANCH" "$WT_DIR" origin/main \
    || die "git worktree add failed"
fi

# Already committed? (resume after step 7)
if G log -1 --format=%s | grep -qF "$SUBJECT"; then
  hdr "already done"
  say "HEAD already carries the re-seal commit."
  say "  branch: $OP_BRANCH"
  say "  commit: $(G rev-parse HEAD)"
  say "Nothing to do. Hand the branch to a session to push and merge."
  exit 0
fi

[ -f "$WT_DIR/$RATIFY_REL" ] || die "$RATIFY_REL is not on origin/main yet: the prepared branch (sub/v7-repin-20260916) must be merged first"

# ------------------------------------------------------------------ 3. dry-run
hdr "3/7 dry-run (writes nothing)"
DRY_LOG="$(mktemp)"; trap 'rm -f "$DRY_LOG"' EXIT
set +e
ROOT="$WT_DIR" EPYC_ROOT="$EPYC_ROOT" LIVE="$LIVE" bash "$WT_DIR/$RATIFY_REL" --dry-run 2>&1 | tee "$DRY_LOG"
dry_rc="${PIPESTATUS[0]}"
set -e
[ "$dry_rc" -eq 0 ] || die "the dry-run refused (exit $dry_rc); read the output above. Nothing written."
ALREADY=0
grep -q '^ALREADY RATIFIED' "$DRY_LOG" && ALREADY=1

# ------------------------------------------------------------------ 4. confirm
if [ "$ALREADY" -eq 0 ]; then
  hdr "4/7 confirm"
  if [ "$YES" -eq 1 ]; then
    say "(test mode: --yes, prompt skipped)"
  else
    [ -r /dev/tty ] || die "no terminal to read the confirmation from; run this interactively"
    printf 'The diff above re-seals the v7 runner pins (6dea92dd and 79721927 -> 20a97fbd)\n'
    printf 'in %s, as operator "%s".\nType RESEAL to apply: ' "$WT_DIR" "$OPERATOR"
    IFS= read -r answer < /dev/tty || die "could not read from the terminal"
    [ "$answer" = "RESEAL" ] || die "confirmation was '$answer', not RESEAL. Nothing applied; the worktree is left for --resume."
  fi

  # ---------------------------------------------------------------- 5. apply
  hdr "5/7 apply (re-runs the zero-inference validations; about 3 minutes)"
  RATIFY_OPERATOR="$OPERATOR" ROOT="$WT_DIR" EPYC_ROOT="$EPYC_ROOT" LIVE="$LIVE" \
    bash "$WT_DIR/$RATIFY_REL" --apply \
    || die "ratify --apply failed (it rolls itself back); read the output above"
else
  hdr "4-5/7 skipped: the re-seal is already applied in this worktree (resume)"
fi

# ------------------------------------------------------------------ 6. stage
hdr "6/7 stage exactly the six re-seal paths"
want="$(printf '%s\n' "${COMMIT_PATHS[@]}" | sort)"
if [ -n "$(G diff --cached --name-only)" ]; then
  staged_pre="$(G diff --cached --name-only | sort)"
  [ "$staged_pre" = "$want" ] || die "the index already holds other staged changes:\n$staged_pre"
fi
G add -- "${COMMIT_PATHS[@]}"
G diff --cached --stat
staged="$(G diff --cached --name-only | sort)"
n="$(printf '%s\n' "$staged" | sed '/^$/d' | wc -l | tr -d ' ')"
[ "$n" = "${#COMMIT_PATHS[@]}" ] || die "staged $n files, expected exactly ${#COMMIT_PATHS[@]}; nothing committed. Inspect: git -C $WT_DIR status"
[ "$staged" = "$want" ] || die "the staged set differs from the six re-seal paths; nothing committed:\n$staged"
other="$(G status --porcelain --untracked-files=all | grep -v '^[AM]  ' || true)"
[ -z "$other" ] || say "note: unstaged or untracked leftovers are NOT committed:"$'\n'"$other"

# ------------------------------------------------------------------ 7. commit
hdr "7/7 commit"
MSG="$(mktemp)"
cat > "$MSG" <<EOF
$SUBJECT

Operator decision 2026-09-16 (RE-SEAL), applied through
scripts/operator/ratify_v7_runner_repin_20260916.sh. All pins were verified
and the section-5 consolidated receipt returned RATIFIED.

- dflash2_followups EXPECTED.runner_sha256 (and the test literal):
  6dea92dd -> 20a97fbd. This crosses da06b371, which adds an opt-in
  --belief-category flag and nothing else.
- P3 bake-off manifest, capture.runner and livecodebench_hard
  scorer.runner_sha256 (+ .sha256 sidecar): 79721927 -> 20a97fbd. This
  crosses baf36757 and da06b371. baf36757 adds the output keys
  effective_request and meta.sampling_fields_are_requested_not_effective;
  requests and scoring are unchanged.

Equivalence (zero inference, data/v7-runner-repin-20260916): request bytes
are identical in 6/6 stub-server scenarios for both sealed versions, the
negative control detects both mutants, and the offline suites are green.
Banked captures keep their historical hashes.

Operator-applied by $OPERATOR
EOF
G commit -q -F "$MSG" || { rm -f "$MSG"; die "git commit failed (hook?); the re-seal is applied and staged in $WT_DIR. Re-run with --resume once resolved."; }
rm -f "$MSG"

hdr "DONE (not pushed)"
say "  worktree: $WT_DIR"
say "  branch:   $OP_BRANCH"
say "  commit:   $(G rev-parse HEAD)"
say "Hand these to a session to push and merge. This script never pushes."
