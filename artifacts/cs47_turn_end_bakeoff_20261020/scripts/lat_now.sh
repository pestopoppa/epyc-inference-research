#!/bin/bash
# contention-tolerant per-decision CPU cost run (build-role q2, nice 10). args: pinlabel cpus threads
T=/mnt/raid0/llm/tmp/cs-47-20261020; O=$T/lat_now
PL=$1; CP=$2; NT=$3
DL=$(date -u -d "2026-10-21T00:10:00Z" +%s)
for job in "silero x" "smartturn x" "parakeet fp32_fp32,160,decoder" "parakeet fp32_fp32,80,decoder" "parakeet fp32_selfdec,160,decoder"; do
  set -- $job; arm=$1; var=$2; py=/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python; [ $arm = smartturn ] && py=$T/venv/bin/python
  [ $(date -u +%s) -lt $DL ] || break
  HF_HUB_OFFLINE=1 nice -n 10 taskset -c $CP $py -I $T/lat_worker.py --arm $arm --variant $var --repeats 10 --clips 6 --pin-label $PL --aa-pass A1 --threads $NT --deadline-epoch $DL --out $O/lat_${arm}_${var//,/_}_${PL}.json >> $O/log_$PL.txt 2>&1
done
