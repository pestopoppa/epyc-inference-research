# Qwen3.5 122B UD-IQ2_M MI210 Smoke

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/model_admission_throughput/qwen35_122b_iq2m_mi210_smoke_20260719T001535Z`

HTTP: `200`; request wall: `0.860s`; startup ready: `7.125s`.

Output exact: `{"status":"ok","model":"qwen35_122b_iq2m"}`

| Metric | Value |
|---|---:|
| Prompt | `43` tokens at `120.26 t/s` |
| Completion | `22` tokens at `45.37 t/s` |

Cleanup: `server_pid_dead` with final ROCm/KFD proof captured in artifact files.

Observation-grade bounded load/coherence smoke; not a production admission or MEASUREMENT gate.
