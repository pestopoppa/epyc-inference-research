#!/bin/bash
# verify_build_linkage.sh — post-build gate for ggml build paths (NIB2-58a).
#
# Runs verify_ggml_linkage.sh against every ggml binary a build directory
# produced, under that build's own launch recipe (LD_LIBRARY_PATH with the
# binary's directory prepended — the loader honours LD_LIBRARY_PATH before the
# binary's own directory, so the recipe is precisely the property under test).
# A build whose binary resolves another tree's ggml (see
# INC-20260731-ggml-linkage-silent-cpu-fallback) must FAIL here, before any
# measurement: "a landed fix whose entry point is never called" is the failure
# this helper exists to prevent, so every ggml build path on this host ends by
# invoking this script.
#
# WHY THIS SCRIPT IS A LOOP AND NOT ONE INVOCATION: ggml build dirs on this host
# differ in layout — llama.cpp puts binaries and libraries in build/bin,
# qwentts.cpp puts tts-server and its libraries directly in build/. A helper
# that takes <build_dir> and finds the binaries avoids one hand-rolled copy of
# the verification logic per build path (the duplicate-logic failure the
# existing wrappers almost produced).
#
# Static builds are handled deliberately: BUILD_SHARED_LIBS=OFF produces
# binaries with no DT_NEEDED on any libggml*, so verify_ggml_linkage.sh exits 2
# (vacuous). For a proven-static binary that is the CORRECT verdict — there is
# no dynamic linkage to mis-resolve — so it is accepted with a note, but ONLY
# when ldd itself says the binary is not dynamic. Any other exit-2 (moved build
# dir, ldd failure, stripped binary) still fails the build closed.
#
# Usage:
#   verify_build_linkage.sh <build_dir> [expected_tree_root]
#     build_dir          the cmake/ninja build directory just built
#     expected_tree_root defaults to build_dir (tighter than the source tree:
#                        a sibling build dir under the same tree is not "inside")
#   VERIFY_GGML_LINKAGE_SH  override the verifier path (defaults to the sibling
#                        script of this one, so both always stay in lockstep)
# Exit:
#   0 = every ggml binary found passes the linkage check under its launch recipe
#   1 = at least one binary FAILED (wrong-tree resolution or vacuous on a
#       dynamic binary) — do not trust any number produced by this build
#   2 = no ggml binaries found in <build_dir> — the build produced nothing to
#       verify, which is itself a build defect, not a pass
set -uo pipefail

BUILD_DIR="${1:?usage: verify_build_linkage.sh <build_dir> [expected_tree_root]}"
EXPECT="${2:-$BUILD_DIR}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERIFIER="${VERIFY_GGML_LINKAGE_SH:-$HERE/verify_ggml_linkage.sh}"

[ -d "$BUILD_DIR" ] || { echo "FAIL: build dir not found: $BUILD_DIR"; exit 1; }
[ -r "$VERIFIER" ] || { echo "FAIL: ggml linkage verifier not readable: $VERIFIER"; exit 1; }

# ggml binary names across the three kernel families on this host.
BIN_NAMES=(
  llama-server llama-cli llama-bench llama-batched-bench llama-perplexity
  llama-imatrix
  whisper-cli whisper-server
  tts-server
)

# Candidate locations: build/bin (llama.cpp, whisper.cpp) and build root
# (qwentts.cpp). Both spellings are checked because the helper must not depend
# on which generation of ggml the tree runs.
CANDIDATE_DIRS=("$BUILD_DIR/bin" "$BUILD_DIR")

found=0
bad=0
for dir in "${CANDIDATE_DIRS[@]}"; do
  [ -d "$dir" ] || continue
  for name in "${BIN_NAMES[@]}"; do
    bin="$dir/$name"
    [ -x "$bin" ] || continue
    found=$((found+1))
    echo "--- $bin"
    LD_LIBRARY_PATH="$dir:${LD_LIBRARY_PATH:-}" bash "$VERIFIER" "$bin" "$EXPECT"
    rc=$?
    if [ "$rc" -eq 0 ]; then
      echo "  OK   $bin resolves inside $EXPECT"
      continue
    fi
    if [ "$rc" -eq 2 ]; then
      # Vacuous. Accept ONLY if the binary is genuinely static: ldd says so
      # itself, which is distinguishable from every other exit-2 cause.
      if ldd "$bin" 2>&1 | grep -qE 'statically linked|not a dynamic executable'; then
        echo "  OK   $bin is statically linked — no dynamic linkage to mis-resolve (vacuous by construction)"
        continue
      fi
      echo "  FAIL $bin: linkage check was VACUOUS on a dynamic binary — broken build layout?"
    else
      echo "  FAIL $bin: ggml libraries resolve OUTSIDE $EXPECT (rc=$rc)"
    fi
    bad=$((bad+1))
  done
done

if [ "$found" -eq 0 ]; then
  cat <<EOF

FAIL: VACUOUS BUILD — no ggml binaries found under $BUILD_DIR.

    binaries inspected : 0

A build path that ends with nothing to verify is indistinguishable from one
whose build silently produced nothing. This is not a pass: if the build's
binaries live elsewhere, point this helper at the right <build_dir>; if the
build only produced libraries, build (or point at) at least one consumer binary
(whisper-cli, tts-server, llama-server, ...) before trusting any measurement.
EOF
  exit 2
fi

echo
if [ "$bad" -gt 0 ]; then
  echo "FAIL: $bad of $found ggml binary/binary-pairs FAILED linkage. Do NOT trust this build."
  exit 1
fi
echo "PASS: all $found ggml binaries resolve their libraries inside $EXPECT"
exit 0
