#!/usr/bin/env python3
"""Exact ROCm 6.2/gfx90a build; compilation does not touch the GPU."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
TARGET = "gfx90a:xnack-:sramecc+"
FLAGS = ["-std=c++17", "-O2", f"--offload-arch={TARGET}", "-mcode-object-version=5",
         "-ffp-contract=off", "-fno-fast-math"]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rocm", type=Path, default=Path("/opt/rocm"))
    args = parser.parse_args()
    out = args.output.resolve()
    if not str(out).startswith("/mnt/raid0/"):
        parser.error("build artifacts must be under /mnt/raid0")
    out.mkdir(parents=True, exist_ok=True)
    hipcc = args.rocm / "bin/hipcc"
    version = subprocess.check_output([str(hipcc), "--version"], text=True)
    if not re.search(r"HIP version: 6\.2\.", version):
        raise SystemExit("unsupported compiler: ROCm 6.2 required")
    commands = []
    def run(command: list[str], name: str) -> str:
        commands.append(command)
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (out / name).write_text(result.stdout)
        if result.returncode:
            raise SystemExit(f"failed ({result.returncode}): {' '.join(command)}; see {out / name}")
        return result.stdout
    run([str(hipcc), *FLAGS, "-c", str(HERE / "kernels.hip"), "-o", str(out / "kernels.o")], "compile.log")
    run([str(hipcc), *FLAGS, "--genco", str(HERE / "kernels.hip"), "-o", str(out / "kernels.bundle")], "code-object.log")
    run([str(args.rocm / "llvm/bin/clang-offload-bundler"), "--unbundle", "--type=o",
         "--targets=hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-",
         f"--input={out / 'kernels.bundle'}", f"--output={out / 'kernels.hsaco'}"], "unbundle.log")
    dis = run([str(args.rocm / "llvm/bin/llvm-objdump"), "-d", str(out / "kernels.hsaco")], "disassembly.txt")
    notes = run([str(args.rocm / "llvm/bin/llvm-readelf"), "-h", "-n", str(out / "kernels.hsaco")], "code-object-metadata.txt")
    if "v_mfma_f32_16x16x16f16" not in dis:
        raise SystemExit("MFMA instruction absent from emitted object")
    if "gfx90a" not in notes or "amdhsa" not in notes.lower():
        raise SystemExit("exact architecture/code object notes missing")
    run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", str(HERE / "test_contract.cpp"), "-o", str(out / "test_contract")], "host-compile.log")
    run([str(hipcc), *FLAGS, "-c", str(HERE / "test_runtime.hip"), "-o", str(out / "test_runtime.o")], "harness-compile.log")
    run([str(hipcc), *FLAGS, str(out / "test_runtime.o"), str(out / "kernels.o"), "-o", str(out / "test_runtime")], "harness-link.log")
    symbols=run(["nm", "--undefined-only", str(out / "kernels.o")], "undefined-symbols.txt")
    if any(name in symbols for name in ("hipMalloc", "hipFree", "_Znwm", "_Znam", " malloc", " calloc", " realloc")):
        raise SystemExit("allocation symbol reachable from standalone runtime object")
    kernel_resources=[]
    for block in notes.split("  - .agpr_count:")[1:]:
        entry={}
        for key in ("name","group_segment_fixed_size","private_segment_fixed_size","sgpr_count","vgpr_count","wavefront_size"):
            match=re.search(r"\."+key+r":\s+(\S+)",block)
            if match:entry[key]=match.group(1)
        kernel_resources.append(entry)
    if not kernel_resources or any(k.get("wavefront_size")!="64" for k in kernel_resources):
        raise SystemExit("emitted kernel does not prove wave64")
    metadata = {
        "schema": "epyc.exl3.gfx90a.build.v1", "target": TARGET, "wavefront_size": 64,
        "code_object_version": 5, "xnack": "off", "sramecc": "on", "compiler": version,
        "compiler_binary_sha256": digest(args.rocm / "llvm/bin/clang++"), "flags": FLAGS,
        "commands": commands, "source_sha256": {p.name: digest(p) for p in sorted(HERE.iterdir()) if p.is_file()},
        "artifact_sha256": {name: digest(out / name) for name in
            ("kernels.o", "kernels.hsaco", "disassembly.txt", "code-object-metadata.txt", "test_contract", "test_runtime")},
        "mfma_instruction_count": dis.count("v_mfma_f32_16x16x16f16"),
        "execution": "not_run", "performance_eligible": False,
        "kernel_resources":kernel_resources,
        "run_allocation_symbols":[], "hbm_traffic":"not_measured", "occupancy":"not_measured",
    }
    (out / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"build": str(out / "build.json"), "target": TARGET, "execution": "not_run"}))

if __name__ == "__main__":
    main()
