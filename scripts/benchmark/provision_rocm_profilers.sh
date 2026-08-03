#!/bin/bash
# Provision the ROCm 6.2 profiler tooling WITHOUT installing into /opt/rocm.
#
#   bash scripts/benchmark/provision_rocm_profilers.sh          # idempotent
#   bash scripts/benchmark/provision_rocm_profilers.sh --verify  # check only
#
# WHY EXTRACTION AND NOT `apt install`
#   /opt/rocm is a bind mount of the HOST's /opt/rocm-6.2.0, shared by the live GPU
#   servers and by three FROZEN kernel trees running three different ggml
#   generations. `apt-get install rocprofiler` writes into that shared tree and can
#   pull dependency upgrades of ROCm runtime libraries underneath a production
#   server -- the exact silent-breakage mode the LD_LIBRARY_PATH discipline exists
#   to prevent. So the packages are `dpkg -x`-extracted to a private prefix.
#   Nothing outside ${ROOT} is modified. Reversal is `rm -rf ${ROOT}`.
#
# WHY THIS SCRIPT EXISTS AT ALL
#   The tooling was first side-loaded by hand on 2026-08-03. Nineteen megabytes of
#   binaries do not belong in git, but the RECIPE does -- otherwise the host state
#   is unreproducible and an untracked directory looks identical to a provisioned
#   one. This is the same failure the P2-5j placement protocol hit when it was
#   deleted from git and left a seed unexecutable.
#
# VERSION MATCHING IS LOAD-BEARING, NOT INCIDENTAL
#   Every package is pinned to 6.2.0-66, matching the installed ROCm exactly. A
#   mismatched profiler against a 6.2 runtime is a defect generator, and the class
#   of defect (counters that read zero, or silently wrong) is precisely the one this
#   tooling exists to rule out.

set -euo pipefail

ROOT="${ROCM_PROFILERS_ROOT:-/mnt/raid0/llm/tools/rocm-profilers-6.2}"
PREFIX="${ROOT}/opt/rocm-6.2.0"
ROCM_APT="https://repo.radeon.com/rocm/apt/6.2/pool/main"
VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

PKGS=(
  "r/rocprofiler6.2.0/rocprofiler6.2.0_2.0.60200.60200-66~22.04_amd64.deb"
  "r/rocprofiler-dev6.2.0/rocprofiler-dev6.2.0_2.0.60200.60200-66~22.04_amd64.deb"
  "r/roctracer6.2.0/roctracer6.2.0_4.1.60200.60200-66~22.04_amd64.deb"
  "r/rocm-bandwidth-test6.2.0/rocm-bandwidth-test6.2.0_1.4.0.60200-66~22.04_amd64.deb"
  "o/omniperf6.2.0/omniperf6.2.0_2.0.1.60200-66~22.04_amd64.deb"
  "h/hsa-amd-aqlprofile6.2.0/hsa-amd-aqlprofile6.2.0_1.0.0.60200.60200-66~22.04_amd64.deb"
)

verify() {
  local ok=0 fail=0
  # shellcheck disable=SC1090
  source "${ROOT}/env.sh" 2>/dev/null || { echo "  env.sh missing"; return 1; }
  for t in rocprofv2 rocprof rocm-bandwidth-test; do
    if command -v "$t" >/dev/null 2>&1 && [[ "$(command -v "$t")" == "${PREFIX}"* ]]; then
      echo "  OK      $t"; ok=$((ok+1))
    else
      echo "  MISSING $t"; fail=$((fail+1))
    fi
  done
  [[ -f "${PREFIX}/lib/rocprofiler/metrics.xml" ]] \
    && { echo "  OK      metrics.xml"; ok=$((ok+1)); } \
    || { echo "  MISSING metrics.xml"; fail=$((fail+1)); }
  [[ -f "${ROOT}/usr/lib/x86_64-linux-gnu/libpciaccess.so.0" ]] \
    && echo "  OK      libpciaccess.so.0 (absent from the base container; see .devcontainer/Dockerfile)" \
    || echo "  NOTE    libpciaccess.so.0 not side-loaded -- fine IF the container now ships it"
  echo "  ${ok} present, ${fail} missing"
  return $(( fail > 0 ))
}

if (( VERIFY_ONLY )); then echo "== verify only =="; verify; exit $?; fi

echo "== provisioning ROCm 6.2 profilers into ${ROOT} =="
echo "   (extraction only -- /opt/rocm is never written)"
INSTALLED_VER="$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
if [[ "${INSTALLED_VER}" != 6.2.* ]]; then
  echo "REFUSING: installed ROCm is '${INSTALLED_VER}', these packages are 6.2.0-66."
  echo "A mismatched profiler against a different runtime produces counters that are"
  echo "silently wrong. Re-pin PKGS to the installed version before running this."
  exit 3
fi

DL="$(mktemp -d)"; trap 'rm -rf "$DL"' EXIT
mkdir -p "$ROOT"

for f in "${PKGS[@]}"; do
  n="$(basename "$f")"
  echo "   fetch  $n"
  curl -sSL --retry 3 -o "${DL}/${n}" "${ROCM_APT}/${f}"
  dpkg -x "${DL}/${n}" "$ROOT"
done

# libpciaccess0 is NOT a ROCm package. It was absent from this container entirely,
# which is why the standing advice "add /usr/lib/x86_64-linux-gnu to LD_LIBRARY_PATH"
# did not work -- the library was not there to find. Now also in the devcontainer
# image; side-loaded here so the tooling works on a container that predates that.
if [[ ! -f /usr/lib/x86_64-linux-gnu/libpciaccess.so.0 ]]; then
  echo "   fetch  libpciaccess0 (absent from base container)"
  ( cd "$DL" && apt-get download libpciaccess0 >/dev/null 2>&1 && dpkg -x libpciaccess0_*.deb "$ROOT" )
fi

# rocprofv2 derives its ROCm root from its own argv[0], NOT from $ROCM_PATH, so the
# private prefix needs its own .info/version or the launcher cats a missing file.
mkdir -p "${PREFIX}/.info"
cp /opt/rocm/.info/version "${PREFIX}/.info/version"

# env.sh is tracked in this repo and symlinked into the prefix, so there is exactly
# one canonical copy and it cannot drift from what git says.
if [[ ! -L "${ROOT}/env.sh" ]]; then
  # resolve to the REAL host path, not via /workspace, so the link survives
  # a container whose /workspace mount differs.
  ln -sfn "$(readlink -f "$(dirname "${BASH_SOURCE[0]}")")/rocm_profilers_env.sh" "${ROOT}/env.sh"
fi

echo
echo "== verifying =="
verify
echo
echo "Activate with:  source ${ROOT}/env.sh"
echo "GPU runs remain operator-approved; having the tool authorizes nothing."
