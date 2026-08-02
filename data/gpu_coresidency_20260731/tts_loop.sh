#!/bin/bash
set -uo pipefail
FIFO=/mnt/raid0/llm/tmp/gpu_coresidency/tts_in
LINES=(
"The accelerator is currently running four models at the same time."
"Speech synthesis proceeds while the language model decodes tokens."
"This sentence is being generated under deliberate contention."
"Measurement discipline requires that every claim be reproducible."
"The vision model is describing an image while this audio is produced."
"Transcription, synthesis and generation share a single device."
)
i=0
while [ ! -f /mnt/raid0/llm/tmp/gpu_coresidency/.stop_load ]; do
  echo "${LINES[$((i % ${#LINES[@]}))]}" > "$FIFO"
  echo "tts line=$i queued"
  i=$((i+1))
  sleep 1
done
echo "tts_loop finished after $i lines"
