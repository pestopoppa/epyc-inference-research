#!/bin/bash
# Verify a ggml-based binary loads ITS OWN libraries, not another tree's.
#
# WHY THIS EXISTS (INC-20260731-ggml-linkage-silent-cpu-fallback)
#
# This host sets, in /etc/environment:5 and .devcontainer/devcontainer.json:57 —
#
#   LD_LIBRARY_PATH="/opt/AMD/aocc-compiler-5.0.0/lib:\
#                    /mnt/raid0/llm/llama.cpp/build/bin:\
#                    /mnt/raid0/llm/llama.cpp-dflash/build/bin:/opt/rocm/lib"
#
# The second entry is the FROZEN PRODUCTION KERNEL. Any freshly built ggml binary
# — whisper.cpp, qwentts.cpp, an experimental llama.cpp — resolves libggml-base /
# libggml-cpu / libggml-hip from THAT tree first, because the loader honours
# LD_LIBRARY_PATH before the binary's own directory.
#
# The failure is silent and produces plausible numbers. On 2026-07-31 a HIP-built
# whisper-cli loaded the production CPU-only ggml, found no GPU, and ran
# full-CPU while printing `use gpu = 1`. Had nobody checked, we would have
# published "GPU whisper" numbers that were CPU numbers. That is a measurement
# integrity failure, not a build annoyance: the run completes, the output is
# well-formed, and only the throughput is quietly wrong.
#
# Do NOT "fix" this by editing /etc/environment — production tooling depends on
# finding the production libs there. Fix it per-invocation by prepending the
# project's own build dir, and call this script to PROVE it worked:
#
#   export LD_LIBRARY_PATH="$MY_BUILD/bin:$LD_LIBRARY_PATH"
#   verify_ggml_linkage.sh "$MY_BUILD/bin/whisper-cli" "$MY_BUILD"
#
# Usage:  verify_ggml_linkage.sh <binary> [expected_tree_root]
#         expected_tree_root defaults to the binary's own directory.
# Exit:   0 = every ggml lib resolves inside the expected tree
#         1 = at least one resolves elsewhere  (DO NOT TRUST THE MEASUREMENT)
set -uo pipefail

BIN="${1:?usage: verify_ggml_linkage.sh <binary> [expected_tree_root]}"
EXPECT="${2:-$(cd "$(dirname "$BIN")" && pwd)}"

[ -x "$BIN" ] || { echo "FAIL: $BIN is not executable"; exit 1; }

echo "binary : $BIN"
echo "expect : libraries under $EXPECT"
echo

# ggml splits across libggml-base / libggml-cpu / libggml-hip|cuda / libggml.
# Backends are dlopened at runtime, so ldd alone is necessary but not sufficient;
# we check the linked set here and report the dlopen search dir separately.
BAD=0
FOUND=0
while read -r name arrow path rest; do
  case "$name" in
    libggml*|libwhisper*|libllama*|libmtmd*) ;;
    *) continue ;;
  esac
  [ "$arrow" = "=>" ] || continue
  [ -n "${path:-}" ] || continue
  FOUND=$((FOUND+1))
  case "$path" in
    "$EXPECT"/*) printf "  OK   %-28s -> %s\n" "$name" "$path" ;;
    *)           printf "  BAD  %-28s -> %s\n" "$name" "$path"; BAD=$((BAD+1)) ;;
  esac
done < <(ldd "$BIN" 2>/dev/null)

if [ "$FOUND" -eq 0 ]; then
  echo "  (no ggml/whisper/llama libs in ldd output — statically linked, or ldd failed)"
fi

echo
echo "LD_LIBRARY_PATH order as the loader sees it:"
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | nl -ba | sed 's/^/    /'

if [ "$BAD" -gt 0 ]; then
  cat <<EOF

FAIL: $BAD library/libraries resolve OUTSIDE $EXPECT.

This binary is running another tree's ggml. Any performance number produced now
is attributable to the WRONG BUILD, and a GPU build may silently execute on CPU
while still reporting a GPU device.

Fix:
    export LD_LIBRARY_PATH="$EXPECT/bin:\$LD_LIBRARY_PATH"
then re-run this check before measuring.
EOF
  exit 1
fi

echo
echo "PASS: all linked ggml libraries resolve inside $EXPECT"
echo "NOTE: ggml backends (ggml-hip, ggml-cpu-<arch>) are dlopened at RUNTIME and"
echo "      are not covered by ldd. Confirm the intended device appears in the"
echo "      program's own startup log (e.g. 'Device 0: AMD Instinct MI210') and"
echo "      do not trust a 'use gpu = 1' flag alone — that flag reports what was"
echo "      REQUESTED, not what was loaded."
exit 0
