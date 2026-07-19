# Qwen3.5 122B UD-IQ2_M MI210 Context Bench

Artifact: `/mnt/raid0/llm/epyc-inference-research/data/model_admission_throughput/qwen35_122b_iq2m_mi210_context_20260719T001712Z`

Source: `experimental-v7-refresh-20260716` @ `ed4091266d286045510e498ceb059c209a65aff9`

| Shape | Throughput |
|---|---:|
| `pp512` | `736.962 t/s` |
| `tg512` | `45.057 t/s` |
| `pp512` | `735.964 t/s` |
| `tg512` | `44.941 t/s` |
| `pp2048+tg512` | `180.149 t/s` |
| `pp4096+tg512` | `269.320 t/s` |

Cleanup: manual post-run check found no residual `llama-bench`/`llama-server` process and no KFD PID.

Caveat: wrapper postflight did not write `exit_code.txt`; treat this as observation-grade complete-output evidence, not a clean canonical measurement.
