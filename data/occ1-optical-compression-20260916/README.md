# OCC-1 optical context compression — MI210, 2026-09-16 (summaries only; no SQuAD text)

Reader: Qwen3-VL-30B-A3B-Instruct Q4_K_M with the F16 mmproj. It was served by `launch_reader.sh`
(research 0d3ca467) on the champion build `b10301-ef81196d5`, n_ctx 16384, 1 slot.
Suite 261d8ac1eaed: SQuAD dev, 39 chunks, 1165 paired questions per arm, 234 requests.
Driver: `occ1_gpu_driver.py` (research branch `sub/gpu-runner-20260916`, not merged).
The pilot (3 chunks) is under `pilot/`.

`records.digest.json` and `plan.digest.json` drop all model output, answers and the SQuAD context and
questions; the source sha256 is kept. The raw run stays at `/mnt/raid0/llm/tmp/occ1-run-20260916`.
