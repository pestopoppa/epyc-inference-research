#!/bin/bash
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
B=/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin
export LD_LIBRARY_PATH=$B
MD=/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf
model_for() { case "$1" in 1017) echo $MD/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf;; sept-q4) echo $MD/DeepSeek-V4.1-Flash-Q4.gguf;; *) exit 9;; esac; }
