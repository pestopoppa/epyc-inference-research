# AKX-SOLSCORE-1: retroactive gap-closed column over the keep ledger (2026-11-22)

Descriptive only; never a promotion input. Zero GPU, read-only (`experiments.db` opened `mode=ro`,
GGUF headers only, bandwidth census read not edited). Classification: OBSERVATION.
Tool: `autokernel/sol_score.py retro`.

| item | value |
|---|---|
| keep rows found (`status='kept'`, all campaign stores) | 22, all CPU (campaign `ak-ds41-cpu-decode-20260923`) |
| GPU keep rows with anchor/candidate tok/s in a ledger | 0 (GPU keeps are recorded by commit/gain only: no Tb/Tk) |
| anchor GGUF `DeepSeek-V4.1-Flash-Q4.gguf` | no longer on disk (retired); bytes taken from proxy `DeepSeek-V4.1-Flash-1017-MXFP4-experts-*` (tagged `bytes_source=proxy_gguf`) |
| active bytes/token (proxy) | 12.62 GB (384 routed experts, 6 used; lookup tables one row) |
| achievable BW | 259.8 GB/s (census `cpu.half-48t`, measured read-sum) |
| Tsol | 48.6 ms/token (20.6 tok/s) |
| status counts | OK 2, BOUND-AUDIT 20, INDETERMINATE 0 |
| rank correlation S vs percent | not reportable (n_scored = 2 < 3) |

Reading: the 20 BOUND-AUDIT rows are the audit working as designed. These are speculative-decode
serving arms (DSpark, draft n=2) whose tok/s counts accepted draft tokens, so the per-token weight
bound (one full pass of active bytes per token) is not a floor for them: Tk < Tsol means the bound
omits acceptance amortisation (and the byte count is a proxy), never that the kernel beat the
hardware. A valid SOL column for spec-decode needs bytes per VERIFY ROUND over accepted tokens per
round, which the keep receipts do not carry.

Open for the done-when: (1) GPU keep rows with absolute Tb/Tk, (2) the retired Q4 GGUF's tensor
table (or a recorded per-round byte count), (3) tokens per round for spec-decode arms. Until those
exist the rank correlation cannot be computed honestly.
