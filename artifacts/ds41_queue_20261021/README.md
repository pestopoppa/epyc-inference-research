# DS41 queue items, prepared 2026-10-21 (no inference was run)

Four DS41-1017 items made runnable and registered in the shared compute queue (`/mnt/raid0/llm/tmp/compute-flags/FILLERS.md`).
All are indicative / correctness-grade and pairable; none is a champion baseline. Host rule: at most ONE DS41-class server at a time.

| id | inner script | analysis |
|---|---|---|
| ds41-c72 (DS41-C7.2) | `c72/run_c72.sh` | `c72/group_nodes.py` (top-k / indexer / lid_kq from a `ggml-cpu-prof/1` node dump) |
| ds41-w61 (DS41-W6.1) | `../ds41_w6_dspark_1017_20261021/w6.1c/run_card.sh` | `w6.1c/analyze_card.py` |
| ds41-t33c (DS41-T3.3c) | `t33c/run_t33c.sh` (+ `t33.py` v3) | `t33c/analyze_t33c.py` (offline, no server) |
| ds41-conf1 (DS41-CONF.1) | `conf1/run_conf1.sh` | `conf1/conf_reliability.py` (llama.cpp patch: branch `experimental/ds41-conf1-logging-20261021` @ b428d7392) |

`lib/grant_launcher.sh` is the generic grant-protocol launcher (poll `grants/<id>.grant`, run only inside its window under a build-role
region-lock claim, TERM/KILL the captured process group at the hard end, `.ended`). `lib/ds41_guard.sh` refuses to start (rc 4) when a
`llama-server` whose argv names a `DeepSeek-V4.1-Flash` model is live (read-only `ps` scan). Launchers run from a frozen copy under
`/mnt/raid0/llm/tmp/ds41-queue-20261021/src`, never from the working tree.
