#!/bin/bash
T=/mnt/raid0/llm/tmp/cs-47-20261020
$T/lat_now.sh pin4 48-51 4 &
$T/lat_now.sh pin1 52 1 &
wait
