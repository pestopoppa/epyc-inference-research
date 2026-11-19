#!/bin/bash
# Build act_dump out-of-tree against a kernel-store build (headers from the store's source commit via git archive).
# Usage: build.sh <store_build_dir> <out_binary> [scratch_dir]
# The store build dir must contain bin/libllama.so + BUILD-IDENTITY.json; headers are taken from its commit
# (read-only `git archive` from /mnt/raid0/llm/llama.cpp, never a checkout). Writes <out_binary>.build.json.
set -euo pipefail
STORE=${1:?store build dir}; OUTB=${2:?output binary}; SCR=${3:-$(dirname "$OUTB")/act_dump-hdr}
HERE=$(cd "$(dirname "$0")" && pwd)
COMMIT=$(/usr/bin/python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("source_sha") or d.get("commit") or d.get("git_commit"))' "$STORE/BUILD-IDENTITY.json")
case "$COMMIT" in ""|None) echo "build.sh: no commit in $STORE/BUILD-IDENTITY.json" >&2; exit 2;; esac
mkdir -p "$SCR"
if [ ! -f "$SCR/include/llama.h" ]; then
  git -C /mnt/raid0/llm/llama.cpp archive "$COMMIT" include ggml/include | tar -x -C "$SCR"
fi
/usr/bin/g++ -std=c++17 -O2 -o "$OUTB" "$HERE/act_dump.cpp" -I"$SCR/include" -I"$SCR/ggml/include" \
  -L"$STORE/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$STORE/bin"
SRC_SHA=$(sha256sum "$HERE/act_dump.cpp" | cut -d' ' -f1)
printf '{"store":"%s","commit":"%s","src_sha256":"%s","bin_sha256":"%s"}\n' "$STORE" "$COMMIT" "$SRC_SHA" \
  "$(sha256sum "$OUTB" | cut -d' ' -f1)" > "$OUTB.build.json"
echo "built $OUTB ($COMMIT)"
