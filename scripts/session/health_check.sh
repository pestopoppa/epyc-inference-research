#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

tmp_dir="${TMPDIR:-/tmp}/epyc-inference-research-health"
mkdir -p "$tmp_dir"

echo "== epyc-inference-research health =="
echo "repo: $(pwd)"
echo "head: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo
echo "== toolchain =="
command -v uv >/dev/null
uv --version

echo
echo "== required paths =="
for path in \
  pyproject.toml \
  Makefile \
  orchestration/model_registry.yaml \
  docs/data/clean_window_measurement_manifest.json \
  docs/data/aa_omniscience_measurement_manifest.json \
  handoffs/active/master-handoff-index.md
do
  test -e "$path"
  echo "ok $path"
done

echo
echo "== python smoke =="
uv run python -m py_compile \
  scripts/research/xmas_winner_table.py \
  scripts/research/xmas_function_axis_sweep.py \
  scripts/benchmark/aa_omniscience_manifest.py \
  scripts/benchmark/clean_window_manifest.py \
  scripts/validate_model_registry.py \
  scripts/validate/check_evidence_durability.py \
  scripts/autopilot/candidate_eval_gate.py \
  scripts/halo/closed_loop_observation_surface.py \
  scripts/halo/convert_tap_to_otel.py

bash -n scripts/session/emergency_cleanup.sh

echo
echo "== registry evidence durability =="
# Every evidence path the registry cites must still RESOLVE on this host: no scratch
# roots, nothing pointing at a file that is gone. Reads one YAML, stdlib only, ~50 ms —
# and it is the check that turns "the artifact behind this ratified hash was swept" from
# a silent event into a red health check. Deliberately NOT a check that the evidence is
# committed: raw campaign output is gitignored on purpose (2026-08-03 operator ruling).
uv run python scripts/validate/check_evidence_durability.py

echo
echo "== no-inference manifest dry-runs =="
uv run python scripts/benchmark/aa_omniscience_manifest.py \
  --dry-run \
  --sample-size 2 \
  --role frontdoor
uv run python scripts/benchmark/clean_window_manifest.py \
  --dry-run \
  --aa-roles frontdoor \
  --k-mem-roles ingest_long_context \
  --k-rope-roles frontdoor \
  --g5-roles frontdoor \
  --output-root "$tmp_dir/clean_window"

echo
echo "health: ok"
