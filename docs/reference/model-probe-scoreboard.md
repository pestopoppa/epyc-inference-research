# Model Probe Scoreboard Pointer

The canonical model-probe scoreboard lives in:

`/mnt/raid0/llm/epyc-root/docs/reference/model-probe-scoreboard.md`

Do not fork or duplicate the table in this repository. Future model probes must append a row to the canonical root scoreboard and may update this repository only for durable benchmark artifacts, registry metadata, or admission-status changes.

Required row fields:

| Field | Requirement |
|---|---|
| Model / quant | Exact model and quantization label. |
| Device | CPU, MI210, hybrid, or serving stack lane. |
| Prompt / decode speed | Label as observation-grade unless a MEASUREMENT protocol applies. |
| Quality | Include task score or explicit quality blocker; speed-only is not role readiness. |
| Role-ready? | Yes / No / gated, with the binding gate. |
| Evidence | Artifact directory or report path. |

Stopped candidates remain stopped unless a concrete quality, loader, protocol, parser, artifact, or compatibility fix states the reopen hypothesis. This includes the Bonsai/Ternary Bonsai, Nemotron-Diffusion, Nemotron-Nano, and extra vision breadth rows already parked in the canonical scoreboard.
