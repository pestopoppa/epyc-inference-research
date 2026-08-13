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
# RESOLVED 2026-07-31: the two llama.cpp entries were REMOVED from the global
# LD_LIBRARY_PATH (/etc/environment:5, .devcontainer/devcontainer.json:57,
# .devc/overrides.json:27). The old header here said "production tooling depends
# on finding the production libs there" — that premise was FALSE. Every binary in
# both production build dirs carries DT_RUNPATH=$ORIGIN and resolves its own
# siblings; an audit of 3192 ELF objects under /mnt/raid0/llm found exactly two
# consumers without a runpath, both unreferenced March-2026 scratch binaries.
#
# Worse, the global entries actively BROKE the GPU: under the old environment,
# /mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server loaded all seven of its
# libraries from the CPU-only build/bin, and libggml-hip.so.0 was not loaded at
# all — because the CPU tree's libggml.so.0 carries no DT_NEEDED on it. The
# production GPU binary was silently a CPU binary.
#
# This script now guards against RE-INTRODUCING those entries. Third-party ggml
# trees (whisper.cpp, qwentts.cpp) still differ in ggml version, so a launcher that
# builds its own env must still prepend the project's own build dir and call this
# script to PROVE it worked:
#
#   export LD_LIBRARY_PATH="$MY_BUILD/bin:$LD_LIBRARY_PATH"
#   verify_ggml_linkage.sh "$MY_BUILD/bin/whisper-cli" "$MY_BUILD"
#
# NON-VACUITY IS INTRINSIC (2026-08-12). This script used to count matched ldd rows
# into FOUND, and when FOUND was 0 it printed "(no ggml/whisper/llama libs in ldd
# output — statically linked, or ldd failed)" as an INFORMATIONAL line and carried
# on to PASS, exit 0. So `verify_ggml_linkage.sh /bin/true <tree>` printed PASS:
# the strongest possible statement — "no library resolves outside this tree" — made
# from having inspected nothing at all. That is the EMPTY-INPUT vacuous pass, and
# a wrong path, a moved build dir, a stripped/static binary or an ldd that failed
# all render as a clean run.
#
# Two conditions now FAIL, both with exit code 2 so a consumer can tell "wrong
# tree" (1) from "this run proved nothing" (2):
#   * zero ggml/whisper/llama libraries inspected;
#   * libggml-base.so absent from the inspected set — EVERY ggml binary on this
#     host links it (whisper-server, tts-server, llama-server), so its absence
#     means the thing under test is not a ggml binary, however many other rows
#     ldd produced.
# scripts/session/verify_speech_kernels.sh (epyc-root) had to bolt exactly this
# gate on from the outside; a check that must be wrapped to be trustworthy is
# untrustworthy everywhere it is NOT wrapped. The `FAIL:`-prefixed output shape and
# the binary/expect header are preserved so both existing consumers' verdict
# regexes still parse (that wrapper, and autokernel's parse_linkage_report).
#
# Usage:  verify_ggml_linkage.sh <binary> [expected_tree_root]
#         expected_tree_root defaults to the binary's own directory.
# Exit:   0 = every ggml lib resolves inside the expected tree
#         1 = at least one resolves elsewhere  (DO NOT TRUST THE MEASUREMENT)
#         2 = VACUOUS / UNINSPECTABLE: nothing ggml was inspected, so this run is
#             not evidence of anything  (DO NOT TRUST THE MEASUREMENT)
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
CORE=0          # libggml-base.so sightings — the intrinsic non-vacuity witness
while read -r name arrow path rest; do
  case "$name" in
    libggml*|libwhisper*|libllama*|libmtmd*) ;;
    *) continue ;;
  esac
  [ "$arrow" = "=>" ] || continue
  [ -n "${path:-}" ] || continue
  FOUND=$((FOUND+1))
  case "$name" in libggml-base.so*) CORE=$((CORE+1)) ;; esac
  case "$path" in
    "$EXPECT"/*) printf "  OK   %-28s -> %s\n" "$name" "$path" ;;
    *)           printf "  BAD  %-28s -> %s\n" "$name" "$path"; BAD=$((BAD+1)) ;;
  esac
done < <(ldd "$BIN" 2>/dev/null)

echo
echo "LD_LIBRARY_PATH order as the loader sees it:"
echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | nl -ba | sed 's/^/    /'

# THE NON-VACUITY GATE, before any verdict. It runs BEFORE the BAD check on
# purpose: "0 of 0 libraries resolved outside the tree" is not a weaker pass, it is
# not a measurement, and it must not be reported as either.
if [ "$FOUND" -eq 0 ] || [ "$CORE" -eq 0 ]; then
  cat <<EOF

FAIL: VACUOUS CHECK — nothing was inspected, so this run proves nothing.

    ggml/whisper/llama libraries inspected : $FOUND
    libggml-base.so seen                   : $CORE   (every ggml binary links it)

This is NOT a pass with a caveat. Reporting "no library resolves outside
$EXPECT" after inspecting $FOUND libraries would be the strongest possible claim
made from the weakest possible evidence.

Likely causes, in order: $BIN is not a ggml binary; it is statically linked; ldd
failed or was denied; the build directory moved. Point this at the real binary
(e.g. \$BUILD/bin/llama-server) and re-run before measuring anything.
EOF
  exit 2
fi

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
