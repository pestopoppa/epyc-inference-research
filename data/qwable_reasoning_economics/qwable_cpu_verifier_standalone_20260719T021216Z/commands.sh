#!/usr/bin/env bash
set -euo pipefail

# Qwable reasoning-economics dry-run plan generated at 2026-07-19T02:12:50.615423+00:00
# plan.json: /mnt/raid0/llm/epyc-inference-research/data/qwable_reasoning_economics/qwable_cpu_verifier_standalone_20260719T021216Z/plan.json

# arm: standalone_iq4_gpu
# resource_class: gpu_iq4
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf --host 127.0.0.1 --port 19180 --device ROCm0 -ngl 99 -t 96 -c 8192 -fa on -rea off
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin curl -fsS http://127.0.0.1:19180/v1/chat/completions -H 'Content-Type: application/json' --data '{"max_tokens":64,"messages":[{"content":"Return a compact JSON object with keys arm, quant, and role using values \"standalone_iq4_gpu\", \"IQ4_XS\", and \"reasoner\".","role":"user"}],"model":"auto","seed":42,"stream":false,"temperature":0.0,"top_k":1,"top_p":1.0}'
if kill -0 $SERVER_PID 2>/dev/null; then kill $SERVER_PID; wait $SERVER_PID 2>/dev/null || true; fi

# arm: standalone_q8_gpu
# resource_class: gpu_q8
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.Q8_0.gguf --host 127.0.0.1 --port 19190 --device ROCm0 -ngl 99 -t 96 -c 8192 -fa on -rea off
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin curl -fsS http://127.0.0.1:19190/v1/chat/completions -H 'Content-Type: application/json' --data '{"max_tokens":64,"messages":[{"content":"Return a compact JSON object with keys arm, quant, and role using values \"standalone_q8_gpu\", \"Q8_0\", and \"reasoner\".","role":"user"}],"model":"auto","seed":42,"stream":false,"temperature":0.0,"top_k":1,"top_p":1.0}'
if kill -0 $SERVER_PID 2>/dev/null; then kill $SERVER_PID; wait $SERVER_PID 2>/dev/null || true; fi

# arm: strict_iq4_json_gpu
# resource_class: gpu_iq4_strict_json
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf --host 127.0.0.1 --port 19200 --device ROCm0 -ngl 99 -t 96 -c 8192 -fa on -rea off
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin curl -fsS http://127.0.0.1:19200/v1/chat/completions -H 'Content-Type: application/json' --data '{"max_tokens":64,"messages":[{"content":"Return exactly this minified JSON and no markdown: {\"arm\":\"strict_iq4_json_gpu\",\"quant\":\"IQ4_XS\",\"role\":\"reasoner\"}","role":"user"}],"model":"auto","seed":42,"stream":false,"temperature":0.0,"top_k":1,"top_p":1.0}'
if kill -0 $SERVER_PID 2>/dev/null; then kill $SERVER_PID; wait $SERVER_PID 2>/dev/null || true; fi

# arm: strict_iq4_schema_gpu
# resource_class: gpu_iq4_schema_json
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf --host 127.0.0.1 --port 19210 --device ROCm0 -ngl 99 -t 96 -c 8192 -fa on -rea off
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin curl -fsS http://127.0.0.1:19210/v1/chat/completions -H 'Content-Type: application/json' --data '{"json_schema":{"additionalProperties":false,"properties":{"arm":{"enum":["strict_iq4_schema_gpu"],"type":"string"},"quant":{"enum":["IQ4_XS"],"type":"string"},"role":{"enum":["reasoner"],"type":"string"}},"required":["arm","quant","role"],"type":"object"},"max_tokens":64,"messages":[{"content":"Return a JSON object for the Qwable schema gate with arm, quant, and role.","role":"user"}],"model":"auto","seed":42,"stream":false,"temperature":0.0,"top_k":1,"top_p":1.0}'
if kill -0 $SERVER_PID 2>/dev/null; then kill $SERVER_PID; wait $SERVER_PID 2>/dev/null || true; fi

# arm: cpu_iq4_baseline
# resource_class: cpu_iq4
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf --host 127.0.0.1 --port 19220 --device none -ngl 0 -t 96 -c 8192 -fa on -rea off
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin curl -fsS http://127.0.0.1:19220/v1/chat/completions -H 'Content-Type: application/json' --data '{"max_tokens":64,"messages":[{"content":"Return a compact JSON object with keys arm, quant, and role using values \"cpu_iq4_baseline\", \"IQ4_XS\", and \"baseline\".","role":"user"}],"model":"auto","seed":42,"stream":false,"temperature":0.0,"top_k":1,"top_p":1.0}'
if kill -0 $SERVER_PID 2>/dev/null; then kill $SERVER_PID; wait $SERVER_PID 2>/dev/null || true; fi

# arm: scaffold_then_beneficiary_stub
# resource_class: hybrid_stub_iq4
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf --host 127.0.0.1 --port 19230 --device ROCm0 -ngl 99 -t 96 -c 8192 -fa on -rea off
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin curl -fsS http://127.0.0.1:19230/v1/chat/completions -H 'Content-Type: application/json' --data '{"max_tokens":64,"messages":[{"content":"Draft a minimal scaffold plan, then tag the beneficiary path as a stub. Return compact JSON with keys arm, scaffold, and beneficiary.","role":"user"}],"model":"auto","seed":42,"stream":false,"temperature":0.0,"top_k":1,"top_p":1.0}'
if kill -0 $SERVER_PID 2>/dev/null; then kill $SERVER_PID; wait $SERVER_PID 2>/dev/null || true; fi

# arm: verifier_selector_stub
# resource_class: cpu_selector_stub
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf --host 127.0.0.1 --port 19240 --device none -ngl 0 -t 96 -c 8192 -fa on -rea off
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin curl -fsS http://127.0.0.1:19240/v1/chat/completions -H 'Content-Type: application/json' --data '{"max_tokens":64,"messages":[{"content":"Draft a verifier-selector stub and return compact JSON with keys arm, verifier, and selector.","role":"user"}],"model":"auto","seed":42,"stream":false,"temperature":0.0,"top_k":1,"top_p":1.0}'
if kill -0 $SERVER_PID 2>/dev/null; then kill $SERVER_PID; wait $SERVER_PID 2>/dev/null || true; fi
