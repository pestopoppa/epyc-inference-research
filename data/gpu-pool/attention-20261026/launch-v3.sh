#!/bin/bash
set -euo pipefail
packet=/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention
approval=$packet/MAIN-APPROVED-RECOVERY-PACKET
if [[ ! -f $approval || -L $approval ]]; then echo 'MAIN bound approval absent or unsafe' >&2; exit 78; fi
if ! packet_sha=$(sha256sum "$packet/SEAL.sha256"); then echo 'packet hash refused' >&2; exit 78; fi
packet_sha=${packet_sha%% *}
if ! grep -Fxq 'proposal=approved' "$approval"; then echo 'MAIN approval refused' >&2; exit 78; fi
if ! grep -Fxq "packet-sha256=$packet_sha" "$approval"; then echo 'MAIN packet binding refused' >&2; exit 78; fi
if ! (cd "$packet" && sha256sum -c SEAL.sha256); then echo 'seal refused' >&2; exit 78; fi
if [[ ${DRY_RUN:-0} == 1 ]]; then
 exec /bin/bash "$packet/dryrun-v3.sh"
fi
export FILLER_STATE_DIR=/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention/state
export FILLERS_MD=/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention/NO-CANONICAL-QUEUE-WRITES
exec /bin/bash /mnt/raid0/llm/tmp/fillers-3c4ca599/filler_run8.sh --id ni-gpu-sampler-attention-20261026-v3 --class gpu --gpu-quiet shared --role build --no-fold-siblings --gpu-threads 184-191 --min-window 1500 --stop /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention/STOP --workdir /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention --cmd '/bin/bash /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention/job-v3.sh' --complete-cmd 'test -f /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention/SUCCESS'
