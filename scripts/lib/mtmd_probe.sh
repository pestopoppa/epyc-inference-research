#!/bin/bash
# Shared bounded llama-mtmd-cli probe used by sourced shell and Python callers.
# stdout is the binary's merged --version output; the process status is returned unchanged.
set -u

if [[ $# -ne 1 || -z "$1" ]]; then
  printf 'usage: %s CANDIDATE_PATH\n' "${0##*/}"
  exit 2
fi

candidate=$1
if [[ ! -x "$candidate" ]]; then
  exit 126
fi

case "$candidate" in
  */*) candidate_dir=${candidate%/*}; [[ -n "$candidate_dir" ]] || candidate_dir=/ ;;
  *) candidate_dir=. ;;
esac
candidate_dir=$(cd -- "$candidate_dir" 2>/dev/null && pwd -P) || exit 126

# The candidate's own library directory must lead. Drop empty entries and repeated
# copies of that same directory while preserving the order of every other entry.
probe_ld_path=$candidate_dir
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  IFS=: read -r -a inherited_dirs <<< "$LD_LIBRARY_PATH"
  for inherited_dir in "${inherited_dirs[@]}"; do
    [[ -n "$inherited_dir" && "$inherited_dir" != "$candidate_dir" ]] || continue
    probe_ld_path+=":${inherited_dir}"
  done
fi

probe_output=$(mktemp "${TMPDIR:-/tmp}/epyc-mtmd-probe.XXXXXX") || exit 125
trap 'rm -f -- "$probe_output"' EXIT
if LD_LIBRARY_PATH="$probe_ld_path" timeout 20 "$candidate" --version >"$probe_output" 2>&1; then
  probe_status=0
else
  probe_status=$?
fi
cat -- "$probe_output"
exit "$probe_status"
