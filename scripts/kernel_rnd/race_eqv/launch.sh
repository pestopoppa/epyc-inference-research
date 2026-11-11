#!/bin/bash
# race-eqv-gpu-r9-20261111: gate seeded-defect builds on the MI210. See spec.json.
set -euo pipefail
D=/mnt/raid0/llm/tmp/race-eqv-gpu-r9-20261111
cd "$D"
/usr/bin/sha256sum --check --status SOURCE.sha256
exec /bin/bash /mnt/raid0/llm/tmp/compute-flags/bin/rowflow/filler_run15.sh --id race-eqv-gpu-r9-20261111 \
  --class gpu --gpu-quiet exclusive --claim-cpus 184-191 --gpu-threads 184-191 --min-window 3600 \
  --cmd "/usr/bin/python3 -I $D/job.py" \
  --complete-cmd "/usr/bin/python3 -I $D/check_complete.py" \
  --stop "$D/STOP" --workdir "$D"
