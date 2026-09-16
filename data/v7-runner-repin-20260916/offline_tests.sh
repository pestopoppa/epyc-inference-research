#!/bin/bash
# offline_tests.sh — run the v7 runner's own offline test suites against each of the
# three runner versions (79721927 sealed-P3, 6dea92dd sealed-dflash2, 20a97fbd current).
# Each version runs inside a private copy of the CURRENT scripts/benchmark directory, so
# the runner file is the only variable. Zero inference: the suites monkeypatch transport.
#
#   bash offline_tests.sh <research checkout> <workdir>   -> <workdir>/offline_tests.tsv
set -euo pipefail
REPO="${1:?research checkout}"; WORK="${2:?workdir}"
LIVE="${LIVE:-/mnt/raid0/llm/epyc-inference-research}"
SUITES=(test_v7_quality_gate_runner.py test_capture_contract_guard.py test_cj_gpqa_sample.py)
declare -A BLOB=(
  [79721927]=511f921c8abd347b32563cc87d407fc0764d8f8a
  [6dea92dd]=5167f5702dcf218ca500efb3cb98d5a0e07e10ca
  [20a97fbd]=b1c2773881858e4750edbc022c93af6f650a64b9
)
mkdir -p "$WORK"
OUT="$WORK/offline_tests.tsv"
printf 'version\tsuite\texit\tsummary\tfailed_tests\n' > "$OUT"
for v in 79721927 6dea92dd 20a97fbd; do
  d="$WORK/offline-$v"
  rm -rf "$d"; mkdir -p "$d/scripts"
  # Same layout as the repo: some suites resolve paths via parents[2]. The banked
  # (untracked) artifacts are mirrored read-only by symlink from the live checkout.
  cp -r "$REPO/scripts/benchmark" "$d/scripts/benchmark"
  find "$d/scripts/benchmark" -name __pycache__ -prune -exec rm -rf {} +
  ln -s "$LIVE/artifacts" "$d/artifacts"
  git -C "$REPO" cat-file blob "${BLOB[$v]}" > "$d/scripts/benchmark/v7_quality_gate_runner.py"
  got="$(sha256sum "$d/scripts/benchmark/v7_quality_gate_runner.py" | cut -c1-8)"
  [ "$got" = "$v" ] || { echo "blob for $v hashed to $got" >&2; exit 1; }
  for s in "${SUITES[@]}"; do
    set +e
    log="$(cd "$d/scripts/benchmark" && PYTHONDONTWRITEBYTECODE=1 timeout 600 python3 -m pytest -q -p no:cacheprovider "$s" 2>&1)"
    rc=$?
    set -e
    summary="$(printf '%s\n' "$log" | tail -1)"
    failed="$(printf '%s\n' "$log" | sed -n 's/^FAILED \([^ ]*\).*/\1/p' | tr '\n' ' ')"
    printf '%s\t%s\t%s\t%s\t%s\n' "$v" "$s" "$rc" "$summary" "${failed:-}" >> "$OUT"
  done
done
cat "$OUT"
