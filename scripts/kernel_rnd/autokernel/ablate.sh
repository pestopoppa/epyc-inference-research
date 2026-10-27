#!/bin/bash
# INF-70 HARNESS-1 (optional third part): a SELF-VERIFYING, TU-proportional ablation arm.
#
# Recipe from autokernel b808e2d6: mutate the tree, build INCREMENTALLY (no clean), then
# object_diff against the champion's saved object manifest. Two things fall out:
#   * cost is proportional to the TUs the mutation touched, not to the tree -- which is what
#     makes leave-one-out cheap enough to run on EVERY champion rebuild instead of occasionally
#     (the gap that let a lever sit at +1.0% while it measured -1.39% in the stack it shipped in);
#   * the diff must name ONLY the lever's own TUs. If it names others, the mutation was not the
#     ablation you thought it was -- the arm verifies itself before it costs a single bench slot.
#
#   usage: ablate.sh <name> <command that mutates the worktree>
# e.g.   ablate.sh qsplit-off "git revert --no-commit <lever-commit>"
#        ablate.sh qsplit-off "sed -i 's/.../.../' ggml/src/ggml-cpu/iqk/iqk_dispatch.cpp"
#
# Run it in a DETACHED ablation worktree that keeps its own build dir; the first arm there pays
# one full seed build, every arm after it pays only its own TUs. This script assumes the build
# dir already holds the champion objects and that objects.manifest.json describes them.
set -euo pipefail
NAME=$1; shift
H=${ABL_HARNESS:-/mnt/raid0/llm/tmp/inf70/agents/harness1}
W=${ABL_WORKTREE:-/mnt/raid0/llm/worktrees/inf70/harness1}
BD=$W/build-cpu
M=${ABL_MANIFEST:-$H/objects.manifest.json}
# This worktree must be exclusively owned by this arm. Never run in an operator's
# checkout. Refuse pre-existing staged or unstaged changes before installing cleanup.
if ! git -C "$W" diff --quiet -- || ! git -C "$W" diff --cached --quiet --; then
  echo "FATAL: $W is dirty; ablation needs a clean champion state"
  exit 1
fi
restore_arm() {
  local original_rc=$? restore_rc=0 paths
  trap - EXIT HUP INT TERM
  # Only tracked paths changed by this arm; do not reset the tree or remove
  # untracked files, which may contain someone else's artifacts.
  paths=$(mktemp) || {
    echo "FATAL: cannot allocate restore path list" >&2
    if (( original_rc == 0 )); then original_rc=1; fi
    exit "$original_rc"
  }
  { git -C "$W" diff --name-only -z -- && git -C "$W" diff --cached --name-only -z --; } > "$paths" || restore_rc=$?
  if (( restore_rc == 0 )) && [[ -s "$paths" ]]; then
    git -C "$W" restore --source=HEAD --staged --worktree --pathspec-from-file="$paths" --pathspec-file-nul || restore_rc=$?
  fi
  rm -f -- "$paths"
  if (( restore_rc )); then
    echo "FATAL: champion restore failed ($restore_rc)" >&2
    if (( original_rc == 0 )); then original_rc=$restore_rc; fi
  fi
  exit "$original_rc"
}
trap restore_arm EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$H/ablate"
echo "=== ablation $NAME: mutating"
( cd "$W" && eval "$@" )
git -C "$W" diff --stat | tee "$H/ablate/$NAME.diffstat"
echo "=== incremental build (NO clean -- this is the point)"
/usr/bin/time -f "INCREMENTAL_BUILD_SECONDS=%e" \
  /workspace/repos/epyc-orchestrator/scripts/region-lock run --cpu-list 48-95 --role build --tag harness1-abl-$NAME -- \
  taskset -c 48-95 cmake --build "$BD" -j 48 --target llama-server > "$H/ablate/$NAME.build.log" 2>&1
grep -E 'INCREMENTAL_BUILD_SECONDS' "$H/ablate/$NAME.build.log" || tail -3 "$H/ablate/$NAME.build.log"
echo "=== object_diff vs champion manifest (the self-verification)"
python3 "$H/tools/objidentity.py" diffm "$M" "$BD" | tee "$H/ablate/$NAME.objdiff.json"
echo "=== restoring champion state"
# EXIT restores staged and unstaged mutations, including on build/diff failure.
