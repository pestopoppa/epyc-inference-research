#!/bin/bash
# CME-7: run the CL-Bench smoke (exploitable_poker, quick_test: 5 hands x 3 runs) through the
# epyc_local system against a LOCAL OpenAI-compatible endpoint. NOT started by the build.
#
# Needs an inference window: refuses unless WINDOW_CONFIRMED=1. Size the run on the served
# model's prefix-cache hit rate and a context of >= ~128K (CME-5, intake-2168#03) BEFORE launch.
#
#   WINDOW_CONFIRMED=1 CLBENCH_REPO=/path/to/clbench CLBENCH_EPYC_BASE_URL=http://127.0.0.1:PORT/v1 \
#     EPYC_MODEL=<served model id> scripts/benchmark/run_clbench_epyc_local.sh
set -euo pipefail

if [[ "${WINDOW_CONFIRMED:-0}" != "1" ]]; then
  echo "refusing: this run needs a granted inference window (set WINDOW_CONFIRMED=1)" >&2
  exit 3
fi
: "${CLBENCH_REPO:?set CLBENCH_REPO to a CL-Bench clone (SHA 5f8c50eb... per CME-6)}"
: "${CLBENCH_EPYC_BASE_URL:?set CLBENCH_EPYC_BASE_URL to the served endpoint, e.g. http://127.0.0.1:PORT/v1}"
: "${EPYC_MODEL:?set EPYC_MODEL to the served model id}"
TASK="${CLBENCH_TASK:-exploitable_poker}"
SCHEDULE="${CLBENCH_SCHEDULE:-quick_test}"
MAX_CTX="${CLBENCH_MAX_CONTEXT_TOKENS:-120000}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prove the endpoint answers before spending a window on it (the caller must also prove it is
# the intended server: foreign servers answer /models too).
curl -fsS --max-time 10 "${CLBENCH_EPYC_BASE_URL%/}/models" >/dev/null \
  || { echo "endpoint ${CLBENCH_EPYC_BASE_URL} not answering /models" >&2; exit 4; }

"$here/clbench_epyc_local/install_into_clbench.sh" "$CLBENCH_REPO"
cd "$CLBENCH_REPO"
exec clbench run "$TASK" --schedule "$SCHEDULE" --system epyc_local \
  --system.model "$EPYC_MODEL" --system.base_url "$CLBENCH_EPYC_BASE_URL" \
  --system.max_context_tokens "$MAX_CTX"
