#!/bin/bash
# Build and execute the REAL ggml decoders against the pinned conformance vectors.
#
#   bash conformance/harness/run_backend_conformance.sh          # CPU + HIP
#   bash conformance/harness/run_backend_conformance.sh --cpu    # CPU only, no GPU touched
#
# This is what moves a backend row from ASSERTED to VERIFIED in
# conformance/matrices/e8m0-conformance.md. Reading source tells you which branch
# of `#if CUDART_VERSION >= 12080` SHOULD be taken; only running it on the real
# toolchain and the real card tells you which one IS.
#
# READ-ONLY on the frozen production tree: the harnesses #include its headers and
# link nothing into it. Nothing here writes to /mnt/raid0/llm/llama.cpp.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="${LLAMA_CPP_ROOT:-/mnt/raid0/llm/llama.cpp}"
CPU_ONLY=0
[[ "${1:-}" == "--cpu" ]] && CPU_ONLY=1

[[ -d "$LLAMA" ]] || { echo "REFUSING: production tree not found at $LLAMA" >&2; exit 1; }

echo "== CPU harness =="
cc -O2 -I"$LLAMA/ggml/include" -I"$LLAMA/ggml/src" \
   -o "$HERE/e8m0_cpu_harness" "$HERE/e8m0_cpu_harness.c"
# Status is taken from cc directly, never from a downstream pipe stage: a build
# whose failure is swallowed reports a number it never computed.
[[ -x "$HERE/e8m0_cpu_harness" ]] || { echo "REFUSING: CPU harness did not build" >&2; exit 2; }
"$HERE/e8m0_cpu_harness" > "$HERE/result_cpu.json"
echo "  wrote result_cpu.json"

if (( CPU_ONLY )); then
  echo "--cpu: skipping the HIP harness (no GPU touched)."
else
  echo "== HIP harness =="
  if ! command -v hipcc >/dev/null; then
    echo "  hipcc not found — skipping; the HIP row stays ASSERTED"
  else
    hipcc -O2 --offload-arch=gfx90a -DGGML_USE_HIP \
      -I"$LLAMA/ggml/include" -I"$LLAMA/ggml/src" -I"$LLAMA/ggml/src/ggml-cuda" \
      -o "$HERE/e8m0_hip_harness" "$HERE/e8m0_hip_harness.hip" 2>&1 \
      | grep -vE 'libxml2|no version information|nodiscard|hipMemcpy|hipGetDeviceProp|\^~|^ *[0-9]+ \|' || true
    [[ -x "$HERE/e8m0_hip_harness" ]] || { echo "REFUSING: HIP harness did not build" >&2; exit 3; }
    "$HERE/e8m0_hip_harness" > "$HERE/result_hip.json"
    echo "  wrote result_hip.json"
  fi
fi

echo
echo "== comparing executed output against the pinned vectors =="
python3 - "$HERE" <<'PY'
import json, sys
from pathlib import Path
here = Path(sys.argv[1]); vec = here.parent / "vectors"
def pinned(name):
    return {c["code"]: c["bits"] for c in json.loads((vec / f"{name}.json").read_text())["cases"]}
checks, failures = [], []
cpu = json.loads((here / "result_cpu.json").read_text())
for key, contract in (("e8m0_ggml_full", "e8m0_ggml_full"), ("e8m0_ggml_half", "e8m0_ggml_half")):
    exp = pinned(contract)
    for c in cpu[key]:
        checks.append(1)
        if c["bits"] != exp[c["code"]]:
            failures.append(f"cpu/{key} code {c['code']}: ran {c['bits']}, pinned {exp[c['code']]}")
hip_path = here / "result_hip.json"
if hip_path.exists():
    hip = json.loads(hip_path.read_text())
    exp = pinned("e8m0_ggml_full")
    for c in hip["e8m0_device"]:
        checks.append(1)
        if c["bits"] != exp[c["code"]]:
            failures.append(f"hip code {c['code']}: ran {c['bits']}, pinned {exp[c['code']]}")
    print(f"  hip arch={hip['arch']} cudart_version_defined={hip['cudart_version_defined']}")
print(f"  {len(checks)} executed values compared")
if failures:
    print("\nMISMATCH — the frozen tree and the pinned vectors DISAGREE:")
    for f in failures:
        print(f"  {f}")
    print("\nThat is the instrument working. Either a decoder changed or a vector is wrong;")
    print("both need a human. Do not 'fix' the vectors to match without establishing which.")
    sys.exit(4)
print("  ALL MATCH — the executed decoders agree with the pinned vectors")
PY
