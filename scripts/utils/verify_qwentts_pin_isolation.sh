#!/bin/bash
# Guard for INF-41 S-13: qwentts.cpp is a PINNED VERSIONED DEPENDENCY, never a merge.
#
# WHY THIS EXISTS
#
# Operator decision 2026-07-31 (multimodal-pipeline.md S-13): qwentts.cpp
# (ServeurpersoCom, third-party fork) and its ggml fork run indefinitely as a
# separate, independently-versioned tree at /mnt/raid0/llm/qwentts.cpp. It is
# NOT part of the project's own production llama.cpp kernel set
# (production-consolidated-vNN) and MUST NOT be folded into it by a future
# "consolidation" pass — CLAUDE.md's four-step experimental-kernel workflow
# (pull fresh production -> build -> validate -> deploy) governs the project's
# OWN llama.cpp tree; qwentts.cpp was never part of that lineage and merging it
# would silently entangle two ggml generations (llama.cpp ggml 0.16.0 vs
# qwentts.cpp's own forked ggml 0.17.0 at this pin) inside one kernel.
#
# This script checks two independent things, either of which failing means the
# isolation invariant has been violated or the pin has drifted:
#
#   1. PIN INTEGRITY — qwentts.cpp is still on its ratified branch/commit, and
#      the ggml submodule is still the ratified commit. (Branch + binary sha256
#      are already covered by scripts/session/verify_speech_kernels.sh in
#      epyc-root; this script adds the ggml SUBMODULE commit, which that script
#      does not check, plus the isolation check below.)
#   2. ISOLATION — the production llama.cpp tree contains NO qwentts.cpp-derived
#      artifacts (source files unique to qwentts.cpp's tts-server) and no commit
#      in its reachable history mentions a merge/import of qwentts.cpp. Absence
#      of evidence is not proof a merge could never happen, but presence of
#      either signal is proof one already did — which is exactly the state this
#      guard exists to catch before it ships as a "cleanup".
#
# Read-only. Starts no process, builds nothing, modifies neither tree.

set -uo pipefail

QWENTTS_TREE="${QWENTTS_TREE:-/mnt/raid0/llm/qwentts.cpp}"
LLAMA_TREE="${LLAMA_TREE:-/mnt/raid0/llm/llama.cpp}"
RATIFICATION="${RATIFICATION:-/workspace/artifacts/operator/ratify_speech_kernel_freeze_20260731.json}"

RC=0

echo "=== S-13 guard: qwentts.cpp pinned-dependency isolation ==="

if [ ! -d "$QWENTTS_TREE/.git" ]; then
    echo "FAIL: qwentts.cpp tree not found or not a git repo: $QWENTTS_TREE"
    exit 1
fi
if [ ! -r "$RATIFICATION" ]; then
    echo "FAIL: ratification artifact not readable: $RATIFICATION"
    exit 1
fi

want_branch=$(python3 -c "import json;print(json.load(open('$RATIFICATION'))['kernels']['qwentts_cpp']['branch'])")
want_commit=$(python3 -c "import json;print(json.load(open('$RATIFICATION'))['kernels']['qwentts_cpp']['commit'])")
want_ggml_sub=$(python3 -c "import json;print(json.load(open('$RATIFICATION'))['kernels']['qwentts_cpp']['ggml_submodule_commit'])")

echo "--- 1. Pin integrity ---"

have_branch=$(git -C "$QWENTTS_TREE" branch --show-current 2>/dev/null)
have_commit=$(git -C "$QWENTTS_TREE" rev-parse HEAD 2>/dev/null)
have_ggml_sub=$(git -C "$QWENTTS_TREE" submodule status -- ggml 2>/dev/null | sed 's/^[ +-U]//' | awk '{print $1}')

if [ "$have_branch" != "$want_branch" ]; then
    echo "  FAIL branch: on '$have_branch', ratified '$want_branch'"; RC=1
else
    echo "  OK   branch: $have_branch"
fi

if [ "$have_commit" != "$want_commit" ]; then
    echo "  FAIL commit: HEAD is $have_commit, ratified $want_commit"; RC=1
else
    echo "  OK   commit: $have_commit"
fi

if [ -z "$have_ggml_sub" ]; then
    echo "  FAIL ggml submodule: could not read submodule status"; RC=1
elif [ "$have_ggml_sub" != "$want_ggml_sub" ]; then
    echo "  FAIL ggml submodule: $have_ggml_sub, ratified $want_ggml_sub"; RC=1
else
    echo "  OK   ggml submodule: $have_ggml_sub"
fi

if [ -n "$(git -C "$QWENTTS_TREE" status --porcelain 2>/dev/null)" ]; then
    echo "  WARN working tree is DIRTY — patches may be unrecorded again (see the"
    echo "       2026-07-31 freeze commit message: this exact failure mode already"
    echo "       happened once before the ggml submodule pointer was committed)."
fi

echo "--- 2. Isolation from the production llama.cpp tree ---"

if [ ! -d "$LLAMA_TREE/.git" ]; then
    echo "  WARN llama.cpp tree not found at $LLAMA_TREE — isolation check skipped, not passed"
else
    # Files that exist ONLY because qwentts.cpp added a TTS server; llama.cpp
    # mainline (and this project's frozen fork of it) has never carried these.
    # A hit means qwentts.cpp source landed inside the llama.cpp tree.
    markers=(tools/tts-server.cpp src/tts-server.h src/prompt-builder.h)
    found=0
    for marker in "${markers[@]}"; do
        if git -C "$LLAMA_TREE" cat-file -e "HEAD:$marker" 2>/dev/null; then
            echo "  FAIL isolation: $marker exists in the production llama.cpp tree at HEAD"
            found=1
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "  OK   no qwentts.cpp-unique source files found in the llama.cpp tree at HEAD"
    else
        RC=1
    fi

    # A merge or cherry-pick that imported qwentts.cpp would very likely leave a
    # trace in commit messages (its own commits, or a merge commit naming it).
    # This is a best-effort trip-wire, not a proof of absence — silence here
    # does not certify isolation on its own, only the absence of this signal.
    hits=$(git -C "$LLAMA_TREE" log --all --grep='qwentts' -i --oneline 2>/dev/null | wc -l)
    if [ "$hits" -gt 0 ]; then
        echo "  FAIL isolation: $hits commit(s) in llama.cpp's reachable history mention 'qwentts'"
        git -C "$LLAMA_TREE" log --all --grep='qwentts' -i --oneline 2>/dev/null | sed 's/^/         /'
        RC=1
    else
        echo "  OK   no commit in llama.cpp's reachable history mentions 'qwentts'"
    fi
fi

echo
if [ "$RC" -eq 0 ]; then
    echo "PASS: qwentts.cpp remains a pinned, isolated, versioned dependency."
else
    echo "FAIL: pin drift or a production-tree isolation violation detected."
    echo "      Do NOT consolidate qwentts.cpp into production-consolidated-vNN."
    echo "      See multimodal-pipeline.md S-13 for the operator decision this guards."
fi
exit $RC
