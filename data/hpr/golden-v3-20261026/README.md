# HPR golden v3 instrument-equivalence evidence (2026-10-26)

Durable copy of the MAIN-accepted HPR golden v3 run (q1 grant CPUs 24-47, 03:30-09:35Z; native COMPLETE 05:40:57Z).
4 families (8k dense, 8k tail_junction, 64k dense, 64k coarse) x 3 probes (cold, matched_extension, divergent_edit)
x 2 endpoints (chat, native) = 24 fresh captures / 12 exact comparisons; schema epyc.hpr.golden_parity.v1 PASS.

Scope: accepted ONLY as prospective instrument equivalence on the pinned experimental CPU model/settings.
Native contract UNVERIFIED. No historical regrade, no performance claim, no parent HPR closure; 64k holdout stays open.

- `GOLDEN_RESULT.json` sha256 3e0f03f0697f678364a7b8c515158ab377545c7e8ba676b0129357acd15c1ec1
- `audit/` accepted checkpoint, AUDIT.md (sha256 64b436763804175006fd8da78671c31e3fb714c19ef6e9985e1a08d3d517f0c2), owner closeout draft, native SHA256SUMS
- `<family>-<endpoint>/` raw captures, server.log, during_request_samples.json, cleanup.json
- `SHA256SUMS` covers every file in this directory

Native source: `/mnt/raid0/llm/tmp/codex-ni-20261024-hpr-recovery/run-golden-v3/live/golden-out-v3/` (tmp; not durable).

## Not copied (> 5 MB) — sealed by audit/NATIVE-EVIDENCE.SHA256SUMS

| path | bytes | sha256 |
|---|---|---|
| `64k-coarse-native/divergent_edit.raw.json` | 5392996 | `bf2241b9840850bfd14c4c1da03001266ac1241eef6f9b292d0e771440c96261` |
| `64k-coarse-native/matched_extension.raw.json` | 5392910 | `3568a5a229ecfb4ea1e1798a42c0a2b23dc3ceee64b09ebc8385c85618325785` |
| `64k-dense-native/divergent_edit.raw.json` | 5393020 | `9fa92cee639be356aa3ff4d4ae3c56d6068172d90179a22b23a9ceb2e106bba6` |
| `64k-dense-native/matched_extension.raw.json` | 5392901 | `17929e7663a569a3741938de5876a076aace4a4e06caf93f13c1de31e2b86f5b` |
