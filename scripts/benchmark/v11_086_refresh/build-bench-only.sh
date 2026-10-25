#!/bin/bash
set -euo pipefail
readonly tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly base=${V11_REFRESH_STATE:?reviewed isolated state root required}
[[ ${FILLER_ID:-} == codex-ni-v11-086-slow2-build-20261025 ]]
mkdir -p "$base/build-logs-v3"
exec > >(tee -a "$base/build-logs-v3/build.log") 2>&1
finish_build() {
  local rc=$?
  if [[ $rc != 0 ]]; then printf "exit=%s\n" "$rc" > "$base/build-logs-v3/BUILDFAILED"; fi
  printf "exit=%s\n" "$rc" > "$base/build-logs-v3/RESULTREADY"
  touch "$base/BUILD-v3-RESULT-CHANGED.flag"
}
trap finish_build EXIT
source "$tools_dir/grant_gate.sh"
[[ -e "$base/synthetic/COMPLETE" && -e "$base/MAIN-APPROVED-BUILD-v3" ]]
(cd "$base" && sha256sum -c BUILD-PROPOSAL-v3.SHA256)
readonly verify=/workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh
for arm in C B; do
  source "$tools_dir/grant_gate.sh"
  case "$arm" in
    B) original=/mnt/raid0/llm/kernels/builds/cpu-20261022-8110933ba-fa-vkq; version=10405; expected=01327fe9c78671d2d79fe22e437a9e2a178012c14b2d342529e05e33d9693275;;
    C) original=/mnt/raid0/llm/kernels/builds/cpu-20261020-f0c96dc1f; version=10404; expected=3f520a5b6552d390c68e9555706b8ed34c27d61c6d07f60f04ed7a24d1e7e435;;
  esac
  src="$base/src-$arm"; candidate="$base/store-candidates-v3/$arm"; build="$base/bench-build-v3-$arm"
  expected_source=$(cat "$base/synthetic/source-$arm.txt")
  [[ "$(git -C "$src" rev-parse HEAD)" == "$expected_source" ]]
  [[ -z "$(git -C "$src" status --porcelain)" ]]
  git -C "$src" rev-parse HEAD > "$base/build-logs-v3/source-$arm.txt"
  # No overwritten or resumed output can masquerade as a new candidate.
  [[ ! -e "$candidate" && ! -e "$build" ]]
  mkdir -p "$candidate/bin" "$candidate/common"
  cp -a "$original/bin/." "$candidate/bin/"
  cp -a "$original/common/libllama-common-base.a" "$candidate/common/"
  (cd "$original"; find bin -maxdepth 1 -type l -printf "%p %l\n" | sort) > "$candidate/SYMLINKS.before"
  (cd "$candidate"; find bin -maxdepth 1 -type l -printf "%p %l\n" | sort) > "$candidate/SYMLINKS.copied"
  cmp "$candidate/SYMLINKS.before" "$candidate/SYMLINKS.copied"
  # All bin files except the two deliberately rebuilt products, plus static common.
  (cd "$original"; find bin -maxdepth 1 -type f ! -name llama-bench ! -name libllama-bench-impl.so -print0 | sort -z | xargs -0 sha256sum; sha256sum common/libllama-common-base.a) > "$candidate/PRESERVED.sha256"
  (cd "$candidate" && sha256sum -c PRESERVED.sha256)
  [[ $(sha256sum "$candidate/bin/libggml-cpu.so.0.16.0" | cut -d' ' -f1) == "$expected" ]]
  /usr/bin/python3 -B -I "$src/tests/test-llama-bench-params-init.py"
  # Only this standalone two-target build graph is ever configured/built.
  cmake -S "$tools_dir/bench-project" -B "$build" -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
    -DPINNED_SOURCE="$src" -DCANDIDATE="$candidate" -DLIBLLAMA_VERSION="$version" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$base/build-logs-v3/configure-$arm.log" 2>&1
  /usr/bin/python3 -B -I "$tools_dir/compile-parity.py" "$original" "$build" "$src" > "$candidate/COMPILE-PARITY.json"
  cmake --build "$build" --target llama-bench -j "${FILLER_THREADS:?}" > "$base/build-logs-v3/compile-$arm.log" 2>&1
  (cd "$candidate" && sha256sum -c PRESERVED.sha256)
  # Symlinks also remain byte-for-byte identical to the read-only source store.
  diff -u <(cd "$original"; find bin -maxdepth 1 -type l -printf '%p %l\n' | sort) \
          <(cd "$candidate"; find bin -maxdepth 1 -type l -printf '%p %l\n' | sort)
  env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH="$candidate/bin" bash "$verify" "$candidate/bin/llama-bench" "$candidate" > "$candidate/LINKAGE.txt" 2>&1
  sed -n "655,658p" "$src/tools/llama-bench/llama-bench.cpp" > "$candidate/HELP-SOURCE.txt"
  grep -Fq -- "--help" "$candidate/HELP-SOURCE.txt"
  grep -Fq "exit(0);" "$candidate/HELP-SOURCE.txt"
  env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH="$candidate/bin" "$candidate/bin/llama-bench" --help > "$candidate/HELP.txt" 2>&1
  printf "0\n" > "$candidate/HELP.rc"
  grep -Fq -- "  -h, --help" "$candidate/HELP.txt"
  grep -Fq "usage:" "$candidate/HELP.txt"
  readelf -d "$candidate/bin/libllama-bench-impl.so" "$candidate/bin/llama-bench" > "$candidate/DYNAMIC.txt"
  printf 'base_store=%s\nsource_commit=%s\nfix=141e4bcf5ae023bd3b871edcd32ad493e6c4d58e\narm=%s\n' "$original" "$(git -C "$src" rev-parse HEAD)" "$arm" > "$candidate/PROVENANCE.txt"
  (cd "$candidate"; find bin -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  # Recheck originals as well, to prove no store mutation happened.
  (cd "$original" && sha256sum -c "$candidate/PRESERVED.sha256")
  (cd "$original"; find bin -maxdepth 1 -type l -printf "%p %l\n" | sort) > "$candidate/SYMLINKS.original-after"
  (cd "$candidate"; find bin -maxdepth 1 -type l -printf "%p %l\n" | sort) > "$candidate/SYMLINKS.candidate-after"
  cmp "$candidate/SYMLINKS.before" "$candidate/SYMLINKS.original-after"
  cmp "$candidate/SYMLINKS.before" "$candidate/SYMLINKS.candidate-after"
done
(cd "$base" && sha256sum -c BUILD-PROPOSAL-v3.SHA256)
printf 'BUILD_ONLY_COMPLETE; acceptance depth launches outstanding\n' > "$base/build-logs-v3/COMPLETE"
