# ROCm 6.2 profiler tooling — side-loaded, NOT installed.
#
# CANONICAL COPY. /mnt/raid0/llm/tools/rocm-profilers-6.2/env.sh is a SYMLINK to this
# file, so there is exactly one copy and it cannot drift from what git says.
# Provision or re-provision the tooling with:
#   bash scripts/benchmark/provision_rocm_profilers.sh
#
#   source /mnt/raid0/llm/tools/rocm-profilers-6.2/env.sh
#
# WHY IT LIVES HERE AND NOT IN /opt/rocm
#   /opt/rocm is a bind mount of the HOST's /opt/rocm-6.2.0, shared by four live
#   GPU servers and three FROZEN kernel trees running three different ggml
#   generations. `apt-get install rocprofiler` would write into that shared tree
#   and could pull dependency upgrades of ROCm runtime libraries underneath a
#   production server — the exact silent-breakage mode CLAUDE.md's LD_LIBRARY_PATH
#   discipline exists to prevent.
#
#   So the packages were EXTRACTED (`dpkg -x`), never installed. Nothing outside
#   this directory was modified. Reversal is `rm -rf` on this directory.
#
# WHAT IS HERE (all version-matched to ROCm 6.2.0-66 — a mismatched profiler
# against a 6.2 runtime is a defect generator, so the match is not incidental):
#   rocprofv2, rocprof (v1), rocm-bandwidth-test, omniperf, metrics.xml,
#   libhsa-amd-aqlprofile64.so.1, and libpciaccess.so.0 (absent from this
#   container entirely — it is the library our own handoff already flagged).
#
# TWO PATH QUIRKS, both handled below:
#   - rocprofv2 derives its ROCm root from its own argv[0], not $ROCM_PATH, so
#     `.info/version` is mirrored into this prefix.
#   - `rocm_agent_enumerator` is taken from the real /opt/rocm (it is present there).

_RPROF_ROOT="/mnt/raid0/llm/tools/rocm-profilers-6.2"
_RPROF_PREFIX="${_RPROF_ROOT}/opt/rocm-6.2.0"

export ROCM_PATH="/opt/rocm"
export PATH="${_RPROF_PREFIX}/bin:/opt/rocm/bin:${PATH}"
export LD_LIBRARY_PATH="${_RPROF_PREFIX}/lib:${_RPROF_ROOT}/usr/lib/x86_64-linux-gnu:/opt/rocm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROCP_METRICS="${_RPROF_PREFIX}/lib/rocprofiler/metrics.xml"

# Verified working 2026-08-03:
#   rocprofv2 --version        -> ROCm 6.2.0-66 / ROCProfiler 2.0
#   rocprofv2 --list-counters  -> 465 distinct gfx90a counters across 12 blocks
#   rocm-bandwidth-test -h     -> OK
#
# PER-BLOCK SIMULTANEOUS-COLLECTION LIMITS (gfx90a) — C4 must schedule around these,
# they are the reason a naive "collect everything" pass silently drops counters:
#   SQ 8 · SPI 6 · TCA 4 · TCC 4 · TCP 4 · CPC 2 · CPF 2 · GRBM 2 · TA 2 · TD 2
#
# STANDING CAVEAT, unchanged by this install: rocprof v1's SQ/TA counters read ZERO
# on this box and it aborts at init on graph-enabled builds. Use rocprofv2.
#
# GPU RUNS REMAIN OPERATOR-APPROVED. Having the tool does not authorize profiling a
# live server; production co-residency is a scheduling question owned by whoever
# owns the inference.
