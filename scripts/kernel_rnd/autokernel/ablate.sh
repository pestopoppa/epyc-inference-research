#!/bin/bash
# Self-verifying leave-one-out arm. Run only inside a filler grant; the worktree
# and build belong exclusively to this transaction. EXPECTED is a JSON list of
# all expected changed object paths relative to build-cpu, with no truncation.
# Usage: ablate.sh NAME EXPECTED_JSON 'mutation command'
set -euo pipefail
[ "$#" -eq 3 ] || { echo 'usage: ablate.sh NAME EXPECTED_JSON MUTATION' >&2; exit 64; }
NAME=$1; EXPECT=$2; MUT=$3
[[ $NAME =~ ^[a-zA-Z0-9._-]+$ ]] || exit 64
: "${ABL_WORKTREE:?}" "${ABL_HARNESS:?}" "${ABL_MANIFEST:?}" "${FILLER_CPUS:?}" "${FILLER_ID:?}" "${FILLER_DEADLINE:?}" "${FILLER_END:?}" "${FILLER_STOPFILE:?}"
[[ $FILLER_END =~ ^[0-9]+$ && $FILLER_DEADLINE =~ ^[0-9]+$ ]] && (( FILLER_DEADLINE < FILLER_END )) || exit 64
[[ $FILLER_CPUS =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] || exit 64
W=$(realpath "$ABL_WORKTREE"); H=$(realpath "$ABL_HARNESS"); M=$(realpath "$ABL_MANIFEST"); EXPECT=$(realpath "$EXPECT")
case "$W" in /mnt/raid0/llm/tmp/*|/mnt/raid0/llm/worktrees/*) ;; *) echo 'FATAL: worktree outside experimental locations' >&2; exit 1;; esac
if git -C "$W" symbolic-ref -q HEAD >/dev/null; then echo 'FATAL: detached experimental worktree required' >&2; exit 1; fi
BD=$W/build-cpu
SCRIPT=$(realpath "$0"); IDENT=$(dirname "$SCRIPT")/ablate_identity.py
[ -d "$BD" ] && [ -f "$M" ] && [ -f "$EXPECT" ] && [ -f "$IDENT" ] || exit 64
# The region claim covers the entire transaction, including cleanup.
exec /workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list "$FILLER_CPUS" --role build --tag "abl-$NAME" -- \
  taskset -c "$FILLER_CPUS" bash "$(dirname "$SCRIPT")/ablate_transaction.sh" "$NAME" "$EXPECT" "$MUT"
