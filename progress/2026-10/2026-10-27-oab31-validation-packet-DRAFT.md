# OAB-31 validation packet — draft, 2026-10-27

OAB-31 remains open. Prepared isolated candidate based on research origin/main `08fe38e7ec99b2092e9db084eb274a8319670be6`, applying the existing private OAB-31 patch cleanly, including its corrected previously failing assertion. The candidate remains unintegrated; the correction has not been tested. Prior v5 evidence was 401 passed, one failed, 48 subtests passed in 28.45 seconds. The production growth guard was red; no exception or growth-guard pass is claimed.

Root reviewed and approved focused model-free validation only. The source-pinned row and launcher are under `/mnt/raid0/llm/tmp/refactor-01a16fc3/oab31-cpu-ready-1728/`; native scoped preflight passed before pytest at `dryrun/c9d049972883e1189b03fcbfc7db7b51c8ca8b96d4ef23de/pass.json`, q0 CPUs 0–1, peak RSS 88,508 KiB and descendants drained. Actual row reserves q0 CPUs 0–23, 4 GiB process cap, 260-second pytest timeout, OMP1, and 150 GiB free after output. Scoped preflight does not establish full-row affinity.

Delivered priority NEXT by identity-checked native IPC at 17:47:46 UTC; receipt `relay-receipt.json`, message ID `298f90d5-6ffe-45aa-a1c1-79b968ddc54b`. Journal recorded enqueue; consumption not yet confirmed. No arm, registration, test result, integration, or completion is claimed by this packet.
