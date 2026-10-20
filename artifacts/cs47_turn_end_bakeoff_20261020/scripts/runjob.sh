#!/bin/bash
# args: dummy variant shardfile outfile
T=/mnt/raid0/llm/tmp/cs-47-20261020; P=/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python
export OMP_NUM_THREADS=1 ORT_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$2" = x ]; then exec nice -n 10 $P -I $T/worker.py --arm silero --clips $3 --out $4 >> $T/logs/jobs.log 2>&1
else exec nice -n 10 $P -I $T/worker.py --arm parakeet --variant $2 --clips $3 --out $4 >> $T/logs/jobs.log 2>&1; fi
