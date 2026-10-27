#!/bin/bash
# Internal transaction; invoked by ablate.sh inside its region-lock claim.
set -euo pipefail
NAME=$1; EXPECT=$2; MUT=$3
W=$(realpath "$ABL_WORKTREE"); H=$(realpath "$ABL_HARNESS"); M=$(realpath "$ABL_MANIFEST")
BD=$W/build-cpu
IDENT=$(dirname "$(realpath "$0")")/ablate_identity.py
mkdir -p "$H/ablate"
exec 9>"$BD/.ablate-transaction.lock"
flock -n 9 || { echo 'FATAL: ablation build in use' >&2; exit 1; }
[ ! -e "$BD/.ablate-quarantined" ] || { echo 'FATAL: previous transaction not verified' >&2; exit 1; }
git -C "$W" diff --quiet -- && git -C "$W" diff --cached --quiet -- || { echo 'FATAL: dirty worktree' >&2; exit 1; }
OUT=$H/ablate/$NAME
[ ! -e "$OUT" ] || { echo 'FATAL: result name already exists' >&2; exit 1; }
mkdir "$OUT"
cp -- "$M" "$OUT/reference.json"
cp -- "$EXPECT" "$OUT/expected.json"
M=$OUT/reference.json; EXPECT=$OUT/expected.json
python3 - "$EXPECT" <<'PY_EXPECT'
import json,sys
value=json.load(open(sys.argv[1]))
if not isinstance(value,list) or not value: raise SystemExit('expected object set must be nonempty')
PY_EXPECT
HEAD=$(git -C "$W" rev-parse HEAD)
printf '[]\n' > "$OUT/empty.json"
python3 "$IDENT" "$M" "$BD" "$OUT/empty.json" "$OUT/seed.json" || { echo 'FATAL: seed identity mismatch' >&2; exit 1; }
# Fail closed after mutation starts. The marker survives failed or interrupted
# cleanup and prevents reuse until an operator verifies the build independently.
CPUS=$(python3 - "$FILLER_CPUS" <<'PY'
import sys
s=set()
for part in sys.argv[1].split(','):
 a,_,b=part.partition('-'); a=int(a); b=int(b or a)
 if b<a: raise SystemExit(64)
 s.update(range(a,b+1))
print(len(s))
PY
)
# Keep five minutes before the scheduler's no-new-work deadline for cleanup;
# this is admission headroom, not a claim that a rebuild has a bounded duration.
arm_admit() {
  local now
  now=$(date -u +%s)
  if (( now + 300 >= FILLER_DEADLINE )) || [ -e "$FILLER_STOPFILE" ] ||
     compgen -G '/mnt/raid0/llm/tmp/compute-flags/HOLD_*' >/dev/null; then
    echo 'REFUSE: stopped, held or insufficient cleanup headroom' >&2
    return 75
  fi
}
arm_admit
touch "$BD/.ablate-quarantined"
build() { cmake --build "$BD" -j "$CPUS" --target llama-server; }
restore_arm() {
  local rc=$? cleanup=0 paths
  trap - EXIT HUP INT TERM
  paths=$(mktemp "$OUT/paths.XXXXXX") || cleanup=1
  [ "$(git -C "$W" rev-parse HEAD)" = "$HEAD" ] || cleanup=1
  if (( cleanup == 0 )); then
    { git -C "$W" diff --name-only -z -- && git -C "$W" diff --cached --name-only -z --; } > "$paths" || cleanup=1
    if (( cleanup == 0 )) && [ -s "$paths" ]; then
      git -C "$W" restore --source=HEAD --staged --worktree --pathspec-from-file="$paths" --pathspec-file-nul || cleanup=1
    fi
    rm -f -- "$paths"
  fi
  [ "$(git -C "$W" rev-parse HEAD)" = "$HEAD" ] || cleanup=1
  git -C "$W" diff --quiet -- && git -C "$W" diff --cached --quiet -- || cleanup=1
  # STOP/HOLD must not suppress cleanup; do not launch a rebuild at or past
  # the hard end. If interrupted or too late, quarantine remains in place.
  if (( cleanup == 0 )); then
    if (( $(date -u +%s) >= FILLER_END )); then cleanup=1
    else build > "$OUT/restore.build.log" 2>&1 || cleanup=1; fi
  fi
  if (( cleanup == 0 )); then python3 "$IDENT" "$M" "$BD" "$OUT/empty.json" "$OUT/restored.json" || cleanup=1; fi
  if (( cleanup == 0 )); then rm -f "$BD/.ablate-quarantined"; else
    echo 'FATAL: restore/rebuild/identity failed; build quarantined' >&2
    (( rc != 0 )) || rc=1
  fi
  exit "$rc"
}
trap restore_arm EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
arm_admit
( cd "$W" && eval "$MUT" )
git -C "$W" diff HEAD --stat > "$OUT/diffstat"
arm_admit
build > "$OUT/arm.build.log" 2>&1
python3 "$IDENT" "$M" "$BD" "$EXPECT" "$OUT/arm.json"
