#!/usr/bin/env python3
"""Build a fresh experimental instrument and record original build provenance.

Run under an appropriate build-region claim. No inference or deployment.
The emitted receipt records observed source and output hashes; it is not an
independent attestation or a measurement-policy verdict.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from kvu126_identity import IdentityError, sha256_file


def source_state(source):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"], text=True)
    if dirty:
        raise IdentityError("instrument source checkout is not clean")
    return commit


def build(args):
    source = Path(args.source).resolve(strict=True)
    destination = Path(args.build_dir).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise IdentityError("build directory must be absent and outside the source checkout")
    before = source_state(source)
    if before != args.expected_commit:
        raise IdentityError("source commit differs from predeclaration")
    destination.mkdir(parents=True)
    config = ["cmake", "-S", str(source), "-B", str(destination)] + args.cmake_arg
    compile_cmd = ["cmake", "--build", str(destination), "--target", "llama-server", "-j", str(args.jobs)]
    started = time.time_ns()
    for command in (config, compile_cmd):
        subprocess.run(command, check=True)
    if source_state(source) != before:
        raise IdentityError("source changed during build")
    executable = destination / "bin" / "llama-server"
    dsos = {}
    for name in args.dso:
        if Path(name).name != name:
            raise IdentityError("invalid or duplicate DSO basename")
        library = (destination / "bin" / name).resolve(strict=True)
        if library.parent != destination / "bin" or library.name in dsos:
            raise IdentityError("library must be unique and inside build bin")
        dsos[library.name] = sha256_file(library)
    if not any(x.startswith("libllama") for x in dsos) or not any(x.startswith("libggml") for x in dsos):
        raise IdentityError("declare all required mapped llama and ggml libraries")
    return {"schema": "epyc.kvu126.build_identity.v1", "source_commit": before, "source_tree_clean": True,
            "source_path": str(source), "build_dir": str(destination), "configure_argv": config,
            "build_argv": compile_cmd, "build_started_unix_ns": started, "build_finished_unix_ns": time.time_ns(),
            "binary_sha256": sha256_file(executable), "dso_sha256": dsos,
            "build_env": {key: os.environ.get(key) for key in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--dso", action="append", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    try:
        receipt = build(args)
        with open(args.receipt, "x") as out:
            json.dump(receipt, out, indent=2)
            out.write("\n")
    except (IdentityError, OSError, subprocess.CalledProcessError) as exc:
        print(f"build refused/failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
