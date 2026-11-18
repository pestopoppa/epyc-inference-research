#!/usr/bin/env python3
"""Make a fold-build kernel-store directory a complete store image.

`autokernel.loop.champion_reanchor.find_store_build` (via `_image_defect`) accepts a `cpu-*` store only
when it has VERSION.txt, SHA256SUMS and PROVENANCE.md and no provenance.json. Fold-build rows write
`SHA256SUMS.<arm>` + BUILD-IDENTITY.json + BUILD-OK.json only, so each store needs this last step.

Reads only files already in the store (plus a `--version` query of the store's own binary, which is not
inference, and `git rev-list --count` for the build number when the binary was built without .git and
reports `0 (unknown)`). Writes exactly VERSION.txt, SHA256SUMS, PROVENANCE.md; leaves any that already exist
alone unless --force.

CLI: python3 store_provenance.py --store DIR --repo LLAMA_REPO [--arm c16] [--out-dir ROW/out] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

VERSION_RE = re.compile(r"^version: (\d+) \(([0-9a-f]{7,40})\)\s*$", re.M)
BUILT_RE = re.compile(r"^built with (\S+) (\S+)", re.M)


def render_version(build_number: int, short_sha: str, built_with_line: str) -> str:
    """`version: N (sha)` + `built with ...`, the exact shape `floor_registry._version_info/_compiler` parse."""
    if not BUILT_RE.match(built_with_line):
        raise ValueError(f"not a `built with <ID> <ver> ...` line: {built_with_line!r}")
    text = f"version: {build_number} ({short_sha})\n{built_with_line.strip()}\n"
    if not VERSION_RE.search(text):
        raise ValueError(f"bad version line inputs: {build_number!r} {short_sha!r}")
    return text


def parse_version_output(out: str) -> tuple[int | None, str | None, str | None]:
    """(build_number, short_sha, built_with_line) from `llama-server --version` output; the number and
    sha are None when the binary reports `0 (unknown)` (source tree had no .git)."""
    m = VERSION_RE.search(out)
    b = re.search(r"^built with .*$", out, re.M)
    return (int(m.group(1)), m.group(2), b.group(0)) if m else (None, None, b.group(0) if b else None)


def render_sha256sums(per_arm_text: str) -> str:
    """`sha  bin/x` (SHA256SUMS.<arm>) -> `sha  ./bin/x`, the form production stores use."""
    out = []
    for line in per_arm_text.splitlines():
        if not line.strip():
            continue
        sha, _, path = line.partition("  ")
        if len(sha) != 64 or not path:
            raise ValueError(f"not a sha256sum line: {line!r}")
        out.append(f"{sha}  {path if path.startswith('./') else './' + path}")
    return "\n".join(out) + "\n"


def _opt(argv: list[str], name: str):
    for a in argv:
        if a.startswith("-D" + name + ":"):
            return a.split("=", 1)[1]
    return None


def _cache(cache_text: str, name: str) -> str:
    m = re.search(r"^" + name + r":[A-Z]+=(.*)$", cache_text, re.M)
    return m.group(1) if m else "n/a"


def render_provenance(store_name: str, arm: str, identity: dict, ok: dict, cache_text: str,
                      out_dir: str, version_note: str) -> str:
    argv = identity["configure_argv"]
    cp = identity["content_proof"]
    kind = identity["kind"]
    return f"""# {store_name}: champion fold candidate {arm.upper()} ({kind.upper()}), NOT production, built {identity['built_at']}

PROVENANCE is the builder's act, written after the fact from facts already in this store and in the build row's out/ dir. No field below was measured fresh; nothing was run beyond a `--version` query.

- commit `{identity['source_sha']}` (tree `{identity['source_tree']}`), branch `{identity['branch']}`, parent `{identity['parent']}`
- source: `{identity['source']}`
- built by row `{identity['built_by_row']}` ({identity['build_s']} s build), kind `{kind}`; output `{identity['output']}`
- recipe (BUILD-IDENTITY.json configure_argv): Release, BUILD_SHARED_LIBS=ON, gcc-15/g++-15, GGML_NATIVE={_opt(argv, 'GGML_NATIVE')}, GGML_OPENMP={_opt(argv, 'GGML_OPENMP')}, GGML_HIP={_opt(argv, 'GGML_HIP')}, LLAMA_CURL={_opt(argv, 'LLAMA_CURL')}, GGML_IQK_Q8_0_PREFILL={_opt(argv, 'GGML_IQK_Q8_0_PREFILL')} (MIN_ROWS={_opt(argv, 'GGML_IQK_Q8_0_PREFILL_MIN_ROWS')}), GGML_IQK_FLASH_ATTENTION={_opt(argv, 'GGML_IQK_FLASH_ATTENTION')}, LLAMA_LLGUIDANCE={_opt(argv, 'LLAMA_LLGUIDANCE')}, relocatability: CMAKE_BUILD_RPATH_USE_ORIGIN=ON, CMAKE_BUILD_WITH_INSTALL_RPATH=ON, CMAKE_INSTALL_RPATH=$ORIGIN
- CMakeCache.txt: CMAKE_BUILD_TYPE={_cache(cache_text, 'CMAKE_BUILD_TYPE')}, CMAKE_CXX_COMPILER={_cache(cache_text, 'CMAKE_CXX_COMPILER')}, CMAKE_HIP_ARCHITECTURES={_cache(cache_text, 'CMAKE_HIP_ARCHITECTURES')}, AMDGPU_TARGETS={_cache(cache_text, 'AMDGPU_TARGETS')}
- content proof (BUILD-IDENTITY.json): llg_matcher_compute_mask strings={cp.get('llg_matcher_compute_mask')}, libggml_hip_present={cp.get('libggml_hip_present')}, markers={[(m['lib'], m['count'], m['ok']) for m in cp.get('markers', [])]}, linkage_rc={identity['linkage_rc']}
- libggml-cpu.so sha256 `{ok.get('libggml_cpu_sha256')}`; libggml-hip.so sha256 `{ok.get('libggml_hip_sha256') or '(none)'}` (BUILD-OK.json)
- file hashes: `SHA256SUMS` (same content as `SHA256SUMS.{arm}`, `./`-prefixed paths) in this directory
- version: {version_note}
- row evidence (not copied into the store): `{out_dir}/` linkage-{kind}-*.txt (verify_ggml_linkage.sh output), logs/configure-{kind}.log, logs/build-{kind}.log, BUILD-{kind}.json, SHA256SUMS-{kind}, preflight-{kind}.txt

## Not sourced
- GPU/CPU gate qualification: not part of the build row; see the gate rows' own results.
"""


def _git(repo: str, *a: str) -> str:
    return subprocess.run(["/usr/bin/git", "-C", repo, *a], capture_output=True, text=True, check=True).stdout.strip()


def version_query(store: Path) -> str:
    """Run the store's own `llama-server --version` linking its own ggml (a version query, not inference)."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LD_LIBRARY_PATH": str(store / "bin")}
    p = subprocess.run([str(store / "bin" / "llama-server"), "--version"], env=env, capture_output=True,
                       text=True, timeout=60)
    return p.stdout + p.stderr


def finalize_store(store: str, repo: str, arm: str | None = None, out_dir: str | None = None,
                   force: bool = False) -> dict:
    s = Path(store)
    targets = [s / n for n in ("VERSION.txt", "SHA256SUMS", "PROVENANCE.md")]
    existing = {t.name for t in targets if t.exists()}  # kept unless force: only missing files are written
    if arm is None:
        found = sorted(p.name.split(".", 1)[1] for p in s.glob("SHA256SUMS.*"))
        if len(found) != 1:
            raise ValueError(f"{store}: cannot infer arm from SHA256SUMS.* = {found}; pass arm")
        arm = found[0]
    identity = json.loads((s / "BUILD-IDENTITY.json").read_text())
    ok = json.loads((s / "BUILD-OK.json").read_text())
    full = identity["source_sha"]
    number, short, built = parse_version_output(version_query(s))
    if built is None:
        raise RuntimeError(f"{store}: --version printed no `built with` line")
    if number is None or number == 0 or short is None or not full.startswith(short):
        number = int(_git(repo, "rev-list", "--count", full))
        short = _git(repo, "rev-parse", "--short=9", full)
        note = (f"binary reports `0 (unknown)` (built from a .git-less source copy); number = `git rev-list --count {full}` "
                f"in {repo} = {number}, matching how upstream computes it (production v10 ffc1bac82 -> 10303); `built with` line is the binary's own")
    else:
        note = "from the store's own `llama-server --version`"
    version = render_version(number, short, built)
    sums = render_sha256sums((s / f"SHA256SUMS.{arm}").read_text())
    prov = render_provenance(s.name, arm, identity, ok, (s / "CMakeCache.txt").read_text(),
                             out_dir or f"(row out dir of {identity['built_by_row']})", note)
    for name, body in (("VERSION.txt", version), ("SHA256SUMS", sums), ("PROVENANCE.md", prov)):
        if force or name not in existing:
            (s / name).write_text(body)
    chk = subprocess.run(["/usr/bin/sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=s, capture_output=True, text=True)
    return {"store": store, "arm": arm, "version": version, "sha256sum_c_rc": chk.returncode, "sha_err": chk.stdout[-300:] + chk.stderr[-300:]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--repo", required=True, help="llama.cpp git repo (read-only rev-list/rev-parse)")
    ap.add_argument("--arm")
    ap.add_argument("--out-dir")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    r = finalize_store(a.store, a.repo, a.arm, a.out_dir, a.force)
    print(json.dumps(r))
    return 0 if r["sha256sum_c_rc"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
