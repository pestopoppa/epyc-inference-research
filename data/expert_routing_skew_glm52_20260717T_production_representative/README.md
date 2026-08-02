# expert routing skew glm52 — 20260717T production representative

Measurement evidence migrated out of scratch on **2026-08-02**. The master registry
cited these artifacts at their `/mnt/raid0/llm/tmp/` paths, so a routine sweep of that
directory would have left ratified, production-affecting claims with nothing behind
them. Copied byte-for-byte (sha256-verified both ends); the scratch originals were
left in place.

| | |
|---|---|
| scratch origin | `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T-production-representative` |
| measured (file mtimes, UTC) | 2026-07-17 05:41 .. 2026-07-17 06:32 |
| migrated | 2026-08-02 |
| carried | 6 files, 505,138 bytes |

## Registry claims this backs

`orchestration/model_registry.yaml` — these citations resolve to this directory.
The YAML key path is the stable reference; line numbers are as of 2026-08-02.

- **L7619** &nbsp;`roles.glm_52_ud_iq2m.performance.expert_routing_skew_representative_observation`
  > data/expert_routing_skew_glm52_20260717T_production_representative/production_representative.corpus.manifest.json).
- **L7621** &nbsp;`roles.glm_52_ud_iq2m.performance.expert_routing_skew_representative_observation`
  > data/expert_routing_skew_glm52_20260717T_production_representative/expert-routing-skew.imatrix.gguf.sha256
- **L7746** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/expert_routing_skew_glm52_20260717T_production_representative/production_representative.corpus.txt
- **L7747** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/expert_routing_skew_glm52_20260717T_production_representative/production_representative.corpus.manifest.json
- **L7748** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/expert_routing_skew_glm52_20260717T_production_representative/expert-routing-skew.imatrix.gguf.sha256
- **L7749** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/expert_routing_skew_glm52_20260717T_production_representative/expert-routing-skew.stats.txt
- **L7750** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/expert_routing_skew_glm52_20260717T_production_representative/expert-routing-skew.counts.json
- **L7751** &nbsp;`roles.glm_52_ud_iq2m.performance.evidence`
  > - data/expert_routing_skew_glm52_20260717T_production_representative/expert-routing-skew.counts.md

## Integrity

`SHA256SUMS` lists every carried file, hashed after the copy and compared against the
scratch original. Verify with:

```bash
cd /mnt/raid0/llm/epyc-inference-research && sha256sum -c data/expert_routing_skew_glm52_20260717T_production_representative/SHA256SUMS
```

## Not carried (hash-only)

Too large for this repository, and `*.gguf` is excluded by `.gitignore` policy.
Recorded here so the artifact stays identifiable and the hash stays checkable
against the scratch original:

| file | bytes | sha256 |
|---|---:|---|
| `/mnt/raid0/llm/tmp/expert-routing-skew-glm52-20260717T-production-representative/expert-routing-skew.imatrix.gguf` | 1,131,474,208 | `f4361b62798cd994b3837263af231f73a30a3e4a36fa14028b4ca877236320a1` |

Build INPUTS and raw-stream byproducts, not measurement results — the numbers
derived from them (`*.counts.json`, `*.stats.txt`, `*_results.json`, the arm logs)
are carried here in full. Re-verify a blob against its hash above while the
scratch original still exists; once it is swept the hash is a record, not a check.

