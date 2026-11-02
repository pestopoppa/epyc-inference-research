#!/bin/bash
# Creates new root-owned read-only experimental source trees in tmpfs only.
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then echo 'root required for the new snapshot only' >&2; exit 2; fi
task_root=/dev/shm/kvu19a2-root-source-20261102
archive=/mnt/raid0/llm/tmp/kvu/kvu19a2-source-pair-20261102/source.tar
patch_path=/mnt/raid0/llm/tmp/kvu/kvu19a2-source-pair-20261102/candidate.patch
mkdir -m 0700 "$task_root"
cp -- "$archive" "$task_root/source.tar"
cp -- "$patch_path" "$task_root/candidate.patch"
printf '%s  %s\n' 87e3cdc13b1859a31dfd3ced43c596e6b9653e2194b1ebd0d5eb6772593c04c2 "$task_root/source.tar" | sha256sum -c -
printf '%s  %s\n' 4147d2b65bbb416fdee1bf004d5447238933524bf3c087a69dcf6bbeca73b235 "$task_root/candidate.patch" | sha256sum -c -
mkdir "$task_root/base" "$task_root/candidate"
tar --no-same-owner -xf "$task_root/source.tar" -C "$task_root/base"
tar --no-same-owner -xf "$task_root/source.tar" -C "$task_root/candidate"
patch -d "$task_root/candidate" -p1 --batch < "$task_root/candidate.patch"
printf '%s  %s\n' 878a3640f685e68c6c3d3102c331c43054bfb0bce81f68c3224ad271b8d9d844 "$task_root/base/ggml/src/ggml-cuda/fattn-tile.cuh" | sha256sum -c -
printf '%s  %s\n' 3f79373e68603691630c7dc38703411456cdf5fa12348d07812323568cdb1e82 "$task_root/candidate/ggml/src/ggml-cuda/fattn-tile.cuh" | sha256sum -c -
differences=$(diff -qr "$task_root/base" "$task_root/candidate" || true)
expected="Files $task_root/base/ggml/src/ggml-cuda/fattn-tile.cuh and $task_root/candidate/ggml/src/ggml-cuda/fattn-tile.cuh differ"
if [[ "$differences" != "$expected" ]]; then echo 'source pair contains unexpected differences' >&2; exit 3; fi
chown -hR root:root "$task_root"
chmod -R a-w "$task_root"
chmod 0755 "$task_root"
