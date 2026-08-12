# VOID RUN — 17 rows, ZERO measurements. Superseded by e5-gemma-nomtp-v9-20260812T0818Z.

Aborted by mainA 2026-08-12T08:05Z. Every cell failed at `affinity preflight exited 1`,
not at the model: affinity_preflight.py could not import fold_cpus_to_physical because
gpu_shadow_lane_lease does an absolute `scripts.server.*` import and only its own
directory was on sys.path. Servers started correctly; the driver recorded zeros.

Fixed in epyc-orchestrator efbbbbe9. The valid run of this configuration is
e5-gemma-nomtp-v9-20260812T0818Z (18/18 cells, 0% error, 43/43 requests per cell).

summary.csv rows here are the driver's record of a failure, NOT measurements.
