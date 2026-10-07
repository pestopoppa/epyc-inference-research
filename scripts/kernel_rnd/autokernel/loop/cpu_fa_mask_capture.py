"""Prepare native DS41 capture provenance before a separately governed CPU run.

This helper starts no server and produces no measurement claim. The capture session
must use the copied prompt/recipe and record its prospective write-side claim carrier.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(*, source_root: Path, build_dir: Path, model: Path, recipe_file: Path,
            prompt_file: Path, capture_dir: Path, run_id: str) -> dict:
    source_root, build_dir, model = (p.resolve(strict=True) for p in (source_root, build_dir, model))
    if source_root == Path("/mnt/raid0/llm/llama.cpp").resolve() or not run_id:
        raise ValueError("capture preparation requires an experimental source and run ID")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source_root), *args], text=True).strip()
    if git("diff", "HEAD", "--name-only"):
        raise ValueError("capture source has tracked changes")
    commit = git("rev-parse", "HEAD")
    commit_time = int(git("show", "-s", "--format=%ct", "HEAD"))
    binary, library = build_dir / "bin/llama-server", build_dir / "bin/libllama.so"
    for image in (binary, library):
        if image.stat().st_mtime < commit_time:
            raise ValueError(f"capture image predates its source commit: {image}")
    if b"AUTOKERNEL_DUMP_FA_MASK" not in library.read_bytes():
        raise ValueError("capture library does not contain the dump hook")
    recipe = json.loads(recipe_file.read_text(encoding="utf-8"))
    if not isinstance(recipe, dict):
        raise ValueError("capture recipe must be a native JSON object")
    capture_dir = capture_dir.absolute()
    capture_dir.mkdir(parents=True, exist_ok=True)
    if any(capture_dir.iterdir()):
        raise ValueError("capture directory must be new and empty")
    manifest = {"schema": "epyc.autokernel.ds41_fa_capture.v1", "architecture": "deepseek41",
                "capture_contract": "ds41_real_mask_n2_5_v1", "source_commit": commit,
                "source_root": str(source_root), "build_dir": str(build_dir), "model": str(model),
                "model_sha256": sha256(model), "recipe_sha256": sha256(recipe_file),
                "prompt_sha256": sha256(prompt_file), "run_id": run_id,
                "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "binary_sha256": sha256(binary), "libllama_sha256": sha256(library)}
    for name, original in (("recipe.json", recipe_file), ("prompt.txt", prompt_file)):
        with (capture_dir / name).open("xb") as target:
            target.write(original.read_bytes())
    with (capture_dir / "capture-manifest.json").open("x", encoding="utf-8") as target:
        json.dump(manifest, target, sort_keys=True, indent=2)
        target.write("\n")
    env = {"LD_LIBRARY_PATH": str(build_dir / "bin"), "AUTOKERNEL_DUMP_FA_MASK": str(capture_dir)}
    env.update({f"AUTOKERNEL_FA_MASK_{key.upper()}": manifest[key] for key in
                ("source_commit", "model", "model_sha256", "run_id", "recipe_sha256", "prompt_sha256")})
    return {"manifest": str(capture_dir / "capture-manifest.json"), "launch_env": env}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source_root", "build_dir", "model", "recipe_file", "prompt_file", "capture_dir"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
