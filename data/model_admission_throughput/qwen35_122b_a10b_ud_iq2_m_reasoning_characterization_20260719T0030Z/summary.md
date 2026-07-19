# Qwen3.5-122B-A10B UD-IQ2_M reasoning characterization

- Date: 2026-07-19 UTC
- Binary: `/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server`
- Model: `/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf`
- Device: ROCm0 / MI210; one server at a time; `-fa on`, `-ctk q8_0`, `-ctv q8_0`, `--spec-type none`
- Determinism: `temperature=0`, `top_p=1`, `top_k=1`, `seed=42`; four prompts, `max_tokens=1024`

## Result

**PASS for characterization; policy remains unsuitable at this bound.**

- `reasoning auto`: 0/4 final answers; 4/4 ended with `finish_reason=length` at exactly 1024 completion tokens; all had empty `message.content` and non-empty `reasoning_content`.
- `reasoning off`: 4/4 final answers; 4/4 ended with `finish_reason=stop`; JSON prompts produced valid JSON and the numbered prompt produced six numbered items.
- This distinguishes the prior 384-token observation from a simple short budget: increasing the budget to 1024 still did not reach final content for these prompts. The evidence points to reasoning/template policy or termination behavior under `auto`, not merely the 384-token cap.

## Throughput

| mode | prompt t/s | decode t/s | completion tokens | wall/request |
|---|---:|---:|---:|---:|
| auto | 152.37 avg | 43.94 avg | 1024 each | 23.71 s each |
| off | 155.69 avg | 44.30 avg | 126, 197, 322, 105 | 3.27, 4.85, 7.71, 2.78 s |

Raw responses, prompts, server logs, timings, pre/postflight state, and command are in this directory.

## Cleanup

- Both servers were terminated between arms and on exit.
- Final `rocm-smi --showpids --showmemuse`: `No KFD PIDs currently running`.
- Final `pgrep -af 'llama-server'` contains no llama-server; the only matching text is the unrelated earlyoom command's ignore regex.
- An intermediate post-stop sample showed 58% allocation, but the final cleanup proof shows 0% VRAM, no KFD PID, and no exact `llama-server`/`llama-bench` process; no production process was restarted or touched.
