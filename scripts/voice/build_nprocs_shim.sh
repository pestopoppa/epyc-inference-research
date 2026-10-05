#!/bin/bash
# Build the nprocs LD_PRELOAD shim for the CPU TTS aux service (see nprocs_shim.c).
# Output lives OUTSIDE every frozen tree, at the path aux_services.tts.env.LD_PRELOAD names.
# Idempotent: rebuilds only when the source hash changes; records the hash next to the .so.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)/nprocs_shim.c"
OUT_DIR="${NPROCS_SHIM_OUT_DIR:-/mnt/raid0/llm/cache/shims}" # override only for tests
OUT="$OUT_DIR/nprocs_shim.so"
mkdir -p "$OUT_DIR"
want="$(sha256sum "$SRC" | cut -d' ' -f1)"
if [ -f "$OUT" ] && [ "$(cat "$OUT.src.sha256" 2>/dev/null)" = "$want" ]; then
  echo "up to date: $OUT (src $want)"
  exit 0
fi
tmp="$(mktemp "$OUT_DIR/.nprocs_shim.XXXXXX.so")"
cc -O2 -Wall -Wextra -shared -fPIC -o "$tmp" "$SRC" -ldl
# Prove it interposes before publishing it: a PLT call to get_nprocs() must see 7.
# (Not `nproc`: glibc's sysconf reaches get_nprocs internally, where no preload can.)
got="$(SHIM_NPROCS=7 LD_PRELOAD="$tmp" python3 -c 'import ctypes; l=ctypes.CDLL(None); print(l.get_nprocs(), l.get_nprocs_conf())')"
[ "$got" = "7 7" ] || {
  rm -f "$tmp"
  echo "REFUSING: shim did not interpose (got '$got')" >&2
  exit 1
}
mv -f "$tmp" "$OUT"
echo "$want" >"$OUT.src.sha256"
echo "built: $OUT (src $want)"
