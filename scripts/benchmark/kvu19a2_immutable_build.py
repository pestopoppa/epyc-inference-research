#!/usr/bin/env python3
"""Build the retained source pair from root-owned, node-read-only tmpfs trees."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT = Path("/dev/shm/kvu19a2-root-source-20261102")
OUT = Path("/dev/shm/kvu19a2-immutable-build-v2-20261102")
TOOLS = Path(__file__).resolve().parents[2] / "artifacts/kvu19a2-repeat-20261102/container-toolchain.sha256"
TOOLS_SHA256 = "2b8ad6a514751640eec363b27a3c9dc29fa17e8479a4a2f144bcf80a648c68dd"
TILES = {"base": "878a3640f685e68c6c3d3102c331c43054bfb0bce81f68c3224ad271b8d9d844",
         "candidate": "3f79373e68603691630c7dc38703411456cdf5fa12348d07812323568cdb1e82"}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(root: Path, *, expected_uid: int = 0):
    """Hash every regular source file; reject symlinks and writable non-root owners."""
    rows = []
    for parent, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
        dirs.sort()
        for name in sorted(dirs + files):
            path = Path(parent) / name
            st = path.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise ValueError(f"source symlink: {path}")
            if st.st_uid != expected_uid or st.st_mode & 0o022:
                raise ValueError(f"source ownership or permissions: {path}")
            if stat.S_ISREG(st.st_mode):
                rows.append({"path": str(path.relative_to(root)), "mode": stat.S_IMODE(st.st_mode),
                             "size": st.st_size, "sha256": file_sha(path)})
            elif not stat.S_ISDIR(st.st_mode):
                raise ValueError(f"source special file: {path}")
    if not rows:
        raise ValueError("empty source inventory")
    return rows


def build():
    if os.geteuid() == 0:
        raise ValueError("build must run as unprivileged node, not root")
    for arm, digest in TILES.items():
        source = ROOT / arm
        st = source.stat()
        if st.st_uid != 0 or st.st_mode & 0o022 or file_sha(source / "ggml/src/ggml-cuda/fattn-tile.cuh") != digest:
            raise ValueError("source root or tile identity mismatch")
    OUT.mkdir(exist_ok=False)
    before = {arm: inventory(ROOT / arm) for arm in TILES}
    manifest = OUT / "source-inventory.json"
    manifest.write_text(json.dumps(before, sort_keys=True, separators=(",", ":")) + "\n")
    if file_sha(TOOLS) != TOOLS_SHA256:
        raise ValueError("toolchain catalog changed")
    subprocess.run(["/usr/bin/sha256sum", "-c", "--quiet", str(TOOLS)], check=True)
    env = {"PATH": "/opt/rocm/bin:/usr/bin:/bin", "ROCM_PATH": "/opt/rocm", "HIP_PATH": "/opt/rocm",
           "CXX": "/usr/bin/g++", "CC": "/usr/bin/gcc", "HIPCXX": "/opt/rocm/llvm/bin/clang++",
           "HOME": str(OUT)}
    outputs = {}
    for arm in TILES:
        target = OUT / arm
        configure = ["/usr/bin/cmake", "-S", str(ROOT / arm), "-B", str(target), "-DCMAKE_BUILD_TYPE=Release",
                     "-DGGML_HIP=ON", "-DAMDGPU_TARGETS=gfx90a", "-DLLAMA_BUILD_TESTS=ON",
                     "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TOOLS=OFF"]
        with (OUT / f"{arm}-configure.log").open("xb") as log:
            subprocess.run(configure, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        cache = (target / "CMakeCache.txt").read_text()
        if f"CMAKE_HOME_DIRECTORY:INTERNAL={ROOT / arm}" not in cache or "GGML_HIP:BOOL=ON" not in cache:
            raise ValueError("CMake source or HIP configuration mismatch")
        with (OUT / f"{arm}-build.log").open("xb") as log:
            subprocess.run(["/usr/bin/cmake", "--build", str(target), "--target", "test-backend-ops",
                            "--parallel", "24"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary, hip = target / "bin/test-backend-ops", target / "bin/libggml-hip.so"
        outputs[arm] = {"test_path": str(binary), "test_sha256": file_sha(binary),
                        "hip_path": str(hip), "hip_sha256": file_sha(hip),
                        "cache_sha256": file_sha(target / "CMakeCache.txt"), "configure": configure,
                        "configure_log_sha256": file_sha(OUT / f"{arm}-configure.log"),
                        "build_log_sha256": file_sha(OUT / f"{arm}-build.log")}
    after = {arm: inventory(ROOT / arm) for arm in TILES}
    if before != after:
        raise ValueError("source changed during build")
    if file_sha(TOOLS) != TOOLS_SHA256:
        raise ValueError("toolchain catalog changed during build")
    subprocess.run(["/usr/bin/sha256sum", "-c", "--quiet", str(TOOLS)], check=True)
    receipt = {"schema": "kvu19a2.root-owned-source-build.v1", "source_root": str(ROOT),
               "source_commit": "a0d0ae238e714aaab04e973ef4b5a4b7c9917744",
               "source_archive_sha256": file_sha(ROOT / "source.tar"),
               "patch_sha256": file_sha(ROOT / "candidate.patch"), "tiles": TILES,
               "source_inventory_sha256": file_sha(manifest), "toolchain_catalog_sha256": file_sha(TOOLS),
               "outputs": outputs,
               "scope": "root-owned read-only to build UID, not immutable against root; transitive toolchain not fully captured"}
    (OUT / "build-receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    build()
