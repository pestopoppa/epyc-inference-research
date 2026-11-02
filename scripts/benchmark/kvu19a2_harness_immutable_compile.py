#!/usr/bin/env python3
"""Build the capture harness from a sealed source fd and root-owned headers."""
import fcntl
import json
import os
from pathlib import Path
import subprocess

from kvu19a2_custody import F_GET_SEALS, sealed_copy, sha_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scripts/benchmark/native/kvu19a2_fa_harness.cpp"
BUILD = Path("/dev/shm/kvu19a2-immutable-build-v2-20261102")
TOOLCHAIN = ROOT / "artifacts/kvu19a2-repeat-20261102/container-toolchain.sha256"
SOURCE_SHA = "b9f352a91ea975c682cbe78c2f4dbf9035c6a40e309b84035bfbdaf49b420091"
TOOLCHAIN_SHA = "2b8ad6a514751640eec363b27a3c9dc29fa17e8479a4a2f144bcf80a648c68dd"


def main():
    if sha_file(TOOLCHAIN) != TOOLCHAIN_SHA:
        raise ValueError("toolchain catalog mismatch")
    subprocess.run(["/usr/bin/sha256sum", "-c", "--quiet", str(TOOLCHAIN)], check=True)
    build_receipt = BUILD / "build-receipt.json"
    if not build_receipt.is_file():
        raise ValueError("source-bound pair build receipt missing")
    receipt = json.loads(build_receipt.read_text())
    if receipt.get("schema") != "kvu19a2.root-owned-source-build.v1":
        raise ValueError("unexpected build receipt")
    build_sha = sha_file(build_receipt)
    out = BUILD / "harness"
    out.mkdir(exist_ok=False)
    executable = out / "fa_harness"
    source_fd = sealed_copy(SRC, SOURCE_SHA)
    try:
        seals = fcntl.fcntl(source_fd, F_GET_SEALS)
        cmd = ["/usr/bin/g++", "-std=c++17", "-O2", "-x", "c++", f"/proc/{os.getpid()}/fd/{source_fd}",
               "-I/dev/shm/kvu19a2-root-source-20261102/base/ggml/include",
               f"-L{BUILD}/base/bin", "-lggml", "-lggml-base", "-lggml-cpu", "-o", str(executable)]
        with (out / "compile.log").open("xb") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(source_fd,))
    finally:
        os.close(source_fd)
    subprocess.run(["/usr/bin/sha256sum", "-c", "--quiet", str(TOOLCHAIN)], check=True)
    if sha_file(build_receipt) != build_sha:
        raise ValueError("build receipt changed during compile")
    data = {"schema": "kvu19a2.harness-build.v1", "source_sha256": SOURCE_SHA, "sealed_memfd_flags": seals,
            "executable_sha256": sha_file(executable), "compile_log_sha256": sha_file(out / "compile.log"),
            "pair_build_receipt_sha256": build_sha, "toolchain_catalog_sha256": TOOLCHAIN_SHA,
            "argv": cmd, "scope": "sealed source bytes and root-owned headers; toolchain transitive inputs unpinned"}
    (out / "harness-receipt.json").write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
