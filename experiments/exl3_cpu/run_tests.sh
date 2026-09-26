#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
build_dir="${EXL3_BUILD_DIR:-$(mktemp -d /tmp/exl3-cpu-XXXXXX)}"
mkdir -p "$build_dir"
python3 - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('fixtures')
manifest=json.loads((root/'manifest.json').read_text())
assert len(manifest['fixtures'])==3, 'mandatory real fixture coverage'
for name,digest in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
print('mandatory real fixture hashes verified')
PY
blas_library="${EXL3_BLAS_LIBRARY:-$(python3 - <<'PY'
import pathlib, scipy
files=list((pathlib.Path(scipy.__file__).parent.parent/'scipy.libs').glob('*openblas*.so'))
if len(files)!=1: raise SystemExit('set EXL3_BLAS_LIBRARY to an existing LP64 CBLAS provider')
print(files[0])
PY
)}"
flags=(-std=c++17 -O2 -g -Wall -Wextra -Werror -ffp-contract=off)
if [[ "${EXL3_SANITIZE:-0}" == 1 ]]; then flags+=(-fsanitize=address,undefined -fno-omit-frame-pointer); fi
"${CXX:-g++}" "${flags[@]}" exl3_cpu.cpp test_cpu.cpp -ldl -o "$build_dir/test_cpu"
if [[ "${1:-}" == --build-only ]]; then printf '%s\n' "$build_dir/test_cpu"; exit 0; fi
OPENBLAS_NUM_THREADS=1 "$build_dir/test_cpu" "$PWD/fixtures" "$blas_library" "$@"
