#!/bin/bash
set -euo pipefail
# Rewrite the editable-install .pth so it follows the checkout you are IN.
# Idempotent. Re-run after any `pip install -e .`, which overwrites it.
#
# WHY
# ---
# `pip install -e .` bakes the absolute path of the checkout it ran from into
# `_editable_impl_epyc_orchestrator.pth`. Git worktrees share this venv, so a
# plain `python scripts/foo.py` from a worktree imports the MAIN checkout's
# `src/` while you edit the worktree's. Source-level `Path(__file__)` anchoring
# cannot fix that — every module is correct and still lands in the wrong tree,
# because the wrong tree is what got imported.
#
# This appends `import epyc_worktree_path` to the .pth. Python executes .pth
# lines beginning with `import`, and that module (tracked at the repo root)
# prepends the CWD's checkout when it differs. The baked path stays as the first
# line so the module itself is importable and the main checkout keeps working
# exactly as before.
#
# Usage: bash scripts/setup/install_editable_pth.sh [--check]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PTH="$(ls "${REPO_ROOT}"/.venv/lib/python*/site-packages/_editable_impl_epyc_orchestrator.pth 2>/dev/null | head -1)"
SHIM="import epyc_worktree_path"

if [[ -z "$PTH" ]]; then
  echo "  SKIP no editable .pth found under ${REPO_ROOT}/.venv — nothing to repair"
  exit 0
fi

if grep -qxF "$SHIM" "$PTH"; then
  echo "  ok   ${PTH} already activates the worktree resolver"
  exit 0
fi

if [[ "${1:-}" == "--check" ]]; then
  echo "  WRONG ${PTH} does not activate the worktree resolver"
  exit 1
fi

cp "$PTH" "${PTH}.bak"
# Deduplicate the baked path (pip wrote it twice here) and append the shim.
# The path must remain FIRST so `import epyc_worktree_path` can resolve.
awk 'NF && !seen[$0]++' "$PTH" >"${PTH}.tmp"
printf '%s\n' "$SHIM" >>"${PTH}.tmp"
mv "${PTH}.tmp" "$PTH"
echo "  installed ${PTH}"
cat "$PTH" | sed 's/^/    | /'
