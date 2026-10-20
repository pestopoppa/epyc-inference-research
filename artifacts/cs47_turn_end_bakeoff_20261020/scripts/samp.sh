#!/bin/bash
T=/mnt/raid0/llm/tmp/cs-47-20261020
while [ -f $T/lat_now/RUNNING ]; do { date -u +%T; cat /proc/loadavg; /workspace/repos/epyc-orchestrator/scripts/region-lock status | tr '\n' ';'; echo; } >> $T/lat_now/contention.log; sleep 60; done
