#!/bin/bash
# AK-ASM-T1 GPU correctness harness r3 (782/782 retarget incl. R9 scratch spill; new + regression units) (handoff row ak-asm-t1): retargeted gfx942 -> gfx90a AITER asm code objects run through AITER's own
# op_tests on the MI210, one unit per kernel family, verdict PASS/FAIL with max error from the family's own checks.
# MI210 + host threads 184-191, GPU quiet (exclusive, role bench). --precond-cmd: no launch unless INSTALL_MANIFEST.json matches the
# installed hsa/gfx90a objects (rc!=0 -> .ended rc=5 reason=install-missing:<line>). A hard GPU hang stops the packet (rc 3).
set -euo pipefail
D=/mnt/raid0/llm/tmp/ak-asm-t1-harness-r3-20261106
cd "$D"
/usr/bin/sha256sum --check --status SOURCE.sha256
exec /bin/bash /mnt/raid0/llm/tmp/compute-flags/bin/rowflow/filler_run15.sh --id ak-asm-t1-harness-r3-20261106-r2 \
  --class gpu --gpu-quiet exclusive --claim-cpus 184-191 --role bench --min-window 3600 \
  --gpu-threads 184-191 \
  --precond-cmd "/usr/bin/python3 -I $D/job.py --precond" \
  --cmd "/usr/bin/python3 -I $D/job.py" \
  --complete-cmd "/usr/bin/python3 -I $D/check_complete.py" \
  --stop "$D/STOP" --workdir "$D"
