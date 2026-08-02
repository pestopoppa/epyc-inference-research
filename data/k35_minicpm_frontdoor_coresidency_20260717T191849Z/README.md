# k35 minicpm frontdoor coresidency — 20260717T191849Z

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/k35-minicpm-frontdoor-coresidency-20260717T191849Z` |
| measured (file mtimes, UTC) | 2026-07-17 19:18 .. 2026-07-17 19:20 |
| migrated | 2026-08-02 |
| carried | 11 files, 46,786 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L8979** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.k35_coresidency_observation`
  > k35_coresidency_observation: "2026-07-17 targeted service-policy smoke passed at data/k35_minicpm_frontdoor_coresidency_20260717T191849Z/: MiniCPM-o reasoning-off MI210 server co-resided with the fastest validated MI210 frontdoor Qwen3.6 lane at 66% VRAM, both handled concurrent requests, MiniCPM...
- **L8988** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.evidence`
  > - data/k35_minicpm_frontdoor_coresidency_20260717T191849Z/summary.json
- **L8989** &nbsp;`roles.minicpm_o_45_local_multimodal.performance.evidence`
  > - data/k35_minicpm_frontdoor_coresidency_20260717T191849Z/post_cleanup_verification.txt

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/k35_minicpm_frontdoor_coresidency_20260717T191849Z/SHA256SUMS
```

