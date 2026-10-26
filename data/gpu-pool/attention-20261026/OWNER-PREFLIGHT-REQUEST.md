# T1 request: owner-only real-launcher preflight window before v3 booking

Per ws58 BLOCKERS-AND-DRYRUN-20261026.md section B, new GPU correctness rows require actual DRYRUN log first. Prior v2 pollers/watchers were safely retired after both grants absent + captured UID1000/SIDself/start113324829/exact argv verification. TERM1136691/1136692/1136693/1136694; subsequent ps returned no rows. Prior sealed v2 payloads and native failed originals remain immutable.

Sampler repair already accepted by MAIN. V3 adds only the requested dry-run preflight path/controls and new identities/roots. Neither v3 launcher is armed, no approval marker exists, no runtime preflight/Python/bash-n/GPU/bench has run, no grant is written by worker. MAIN exact hash approval remains needed; MAIN may write regular non-symlink MAIN-APPROVED-RECOVERY-PACKET containing proposal=approved plus packet-sha256=<actual SEAL.sha256 digest> in each v3 family root. Launch and job enforce binding plus all manifest entries.

Missing external event: owner allocation of a real preflight-only window with two CPUs in one free quarter (none free before11:37 per MAIN), or explicit owner GPU-host preflight grant mi210_0,184-191 with two of184-191 used. Runtime cap300 seconds,1 thread,2 CPUs. Reserve enough official grant wall time for300s work plus owner deadline margin. No model/server/native GPU bench. The helper reads the real canonical grant for its own PREFLIGHT_ID before tests; actual affinity must equal the specified pair, which must belong to the real grant. CPU-only grant regions must be exactly that pair; GPU alternative must be exactly mi210_0,184-191. It records PREFLIGHT-AUTHORITY.json. No synthetic authority and no canonical fake grants.

Proposed preflight identities:
- ni-gpu-sampler-attention-pf-20261026-v3
- ni-gpu-sampler-elementwise-pf-20261026-v3

The owner can instead grant the corresponding new GPU correctness ID for an explicitly preflight-only command; the real launcher DRY_RUN branch exits before filler/native launch. No correctness booking may infer a native run from that preflight grant.

Exact entry commands, only inside the owner-created corresponding real grant and after MAIN bound approval (replace OWNER_CPU_A,OWNER_CPU_B with the owner-authorized explicit pair;184,185 only for accepted GPU-host shape):

```sh
PREFLIGHT_ID=ni-gpu-sampler-attention-pf-20261026-v3 PREFLIGHT_CPUS=OWNER_CPU_A,OWNER_CPU_B DRY_RUN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 timeout --kill-after=5s 300s /bin/bash /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention/launch-v3.sh > /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/attention/DRYRUN.log 2>&1
PREFLIGHT_ID=ni-gpu-sampler-elementwise-pf-20261026-v3 PREFLIGHT_CPUS=OWNER_CPU_A,OWNER_CPU_B DRY_RUN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 timeout --kill-after=5s 300s /bin/bash /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/elementwise/launch-v3.sh > /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/elementwise/DRYRUN.log 2>&1
```

The actual launcher's bounded branch checks real grant/abort flags then sealed executable/DSOs/shim/native source/compiled CLI and graph knob; native artifact mtime is compared to its native source commit, never to the later interpreted Python fix. Python fix is hash sealed and compiled. It checks intended LD_LIBRARY_PATH with verify_ggml_linkage.sh, mask expansion184-191, requested8GiB host estimate and1800>=1400+300, portsnone, recipe stack and owned flag/output/state write paths; exercises missing fake grant/HOLD/STOP/wrong shape abort in isolated scratch plus25 meaningful sampler controls. Fake results never serve as residency/native acceptance. Native no-grant runtime fixture path remains unchanged.

After actual DRYRUN success, MAIN may arm new pollers/watchers, send exact ready18-column rows plus actual DRYRUN log to owner, and owner books correctness NOW before quiet night if feasible. No current champion/performance/release/clean-source claim; old partial family records remain partial. Source/BK coverage should extend existing SC18/SC88 to v3 carriers, not create a grading rule.
