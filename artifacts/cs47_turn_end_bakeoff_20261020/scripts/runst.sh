#!/bin/bash
T=/mnt/raid0/llm/tmp/cs-47-20261020
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
exec nice -n 10 $T/venv/bin/python -I $T/worker.py --arm smartturn --clips $1 --out $2 >> $T/logs/jobs_st.log 2>&1
