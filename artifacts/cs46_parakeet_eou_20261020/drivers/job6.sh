#!/bin/bash
D=/mnt/raid0/llm/tmp/cs-46-20261020
$D/rust/build.sh > $D/rust/build2.log 2>&1; echo "[job6] build rc=$?"
$D/job5.sh
