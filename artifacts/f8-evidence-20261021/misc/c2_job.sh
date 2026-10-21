#!/bin/bash
# candidate C2 = C (f0c96dc1f) + CT-12.3b (6dfbc9e86 -> b4dd52370). Builds (CPU, GPU compile) and CPU correctness tests at nice 10.
# Runs inside a build-role region claim. Output: c2/job.log, c2/results.txt
set -uo pipefail
D=/mnt/raid0/llm/tmp/champ-consolidation-20261020/c2
W=/mnt/raid0/llm/worktrees/llama-champ-consolidation-c2-20261021
SHA=$(git -C $W rev-parse --short=9 HEAD)
export FOLD_BUILD_CPUS=72-87,168-183 FOLD_BUILD_JOBS=32
# guard: never run inside the coordinator's CPU timing window (about 11:30Z-14:30Z); if the claim only arrives then, exit and wait for a manual relaunch
NOW=$(date -u +%H%M); if [ "$NOW" -ge 1115 ] && [ "$NOW" -lt 1445 ]; then echo "[$(date -u +%FT%TZ)] inside the timing window, not running" >> $D/job.log; exit 3; fi
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $D/job.log; }
say "C2 sha $SHA"
$D/build_store_c2.sh cpu > $D/build_cpu.out 2>&1; say "cpu build rc=$? $(tail -n1 $D/build_cpu.out)"
$D/build_store_c2.sh gpu > $D/build_gpu.out 2>&1; say "gpu build rc=$? $(tail -n1 $D/build_gpu.out)"
CB=/mnt/raid0/llm/kernels/builds/cpu-20261021-$SHA
: > $D/results.txt
# 1. unit: test-chat-input-spans (needs the vocab file under models/ relative to the worktree root)
( cd $W && env -u LD_LIBRARY_PATH nice -n 10 taskset -c 72-87 $CB/bin/test-chat-input-spans ) > $D/unit_spans.log 2>&1; echo "test-chat-input-spans rc=$? $(tail -n 1 $D/unit_spans.log)" >> $D/results.txt
# 2. pytest test_chat_special_tokens.py (7 expected)
( cd $W/tools/server/tests && rm -rf tmp && mkdir -p tmp && LLAMA_SERVER_BIN_PATH=$CB/bin/llama-server LLAMA_TEST_CHATML_MODEL=/mnt/raid0/llm/models/QuantFactory/Qwen2.5-0.5B-GGUF/Qwen2.5-0.5B.Q8_0.gguf PORT=18975 N_GPU_LAYERS=0 nice -n 10 taskset -c 72-87 /mnt/raid0/llm/epyc-orchestrator/.venv/bin/python -m pytest --noconftest unit/test_chat_special_tokens.py -p no:cacheprovider ) > $D/pytest.log 2>&1; echo "pytest rc=$? $(tail -n 1 $D/pytest.log)" >> $D/results.txt
rm -rf $W/tools/server/tests/tmp
# 3. CT-12.4 template-golden A/B (champion vs C2)
sed "s#@C2SHA@#$SHA#" $D/ct12_4_goldens_ab_c2.sh > $D/ct12_4_run.sh; chmod +x $D/ct12_4_run.sh
nice -n 10 taskset -c 72-87 $D/ct12_4_run.sh > $D/ct12_4.log 2>&1; echo "ct12-4 A/B rc=$? $(grep RESULT $D/ct12_4.log | tail -n 1)" >> $D/results.txt
# 4. regression: existing chat-path unit tests of the tree (A/B vs the champion build is in the CT-12 evidence; here C2 only)
for t in test-chat test-jinja test-chat-template test-chat-auto-parser test-chat-peg-parser; do
  [ -x $CB/bin/$t ] || continue
  ( cd $W && env -u LD_LIBRARY_PATH nice -n 10 taskset -c 72-87 timeout 900 $CB/bin/$t ) > $D/unit_$t.log 2>&1; echo "$t rc=$? $(tail -n 1 $D/unit_$t.log | cut -c1-100)" >> $D/results.txt
done
say "job done"
