#!/usr/bin/env python3
"""Download long-context evaluation datasets to /mnt/raid0/llm/data/eval/.

HF datasets v4.x dropped support for custom loading scripts. LongBench v1,
ZeroSCROLLS, and L-Eval all use deprecated .py loaders. We work around this:

  - LongBench: Use v2 (THUDM/LongBench-v2) which has native parquet support.
    503 multiple-choice questions across long-context tasks.
  - ZeroSCROLLS: Download raw zip files via huggingface_hub and extract JSONL.
  - L-Eval: Download raw zip/jsonl files via huggingface_hub and extract.

RULER and Needle-in-a-Haystack are git repos (cloned separately).

Usage:
    python download_long_context_datasets.py [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path("/mnt/raid0/llm/data/eval")


def _download_longbench_v2(force: bool = False) -> dict:
    """Download THUDM/LongBench-v2 — 503 multiple-choice long-context questions."""
    import datasets as hf

    target = EVAL_DIR / "longbench"
    meta = {"name": "LongBench-v2", "source": "THUDM/LongBench-v2", "license": "MIT"}

    jsonl_path = target / "longbench_v2.jsonl"
    if jsonl_path.exists() and not force:
        lines = jsonl_path.read_text().strip().split("\n")
        meta["total_rows"] = len(lines)
        print(f"  [longbench] Already cached: {len(lines)} examples")
        return meta

    try:
        print("  [longbench] Downloading THUDM/LongBench-v2...")
        ds = hf.load_dataset("THUDM/LongBench-v2", split="train",
                             cache_dir=str(target))
        # Export to JSONL for fast loading
        rows = []
        for row in ds:
            rows.append(json.dumps(row, ensure_ascii=False))
        jsonl_path.write_text("\n".join(rows))
        meta["total_rows"] = len(rows)
        meta["columns"] = ds.column_names
        print(f"  [longbench] {len(rows)} examples saved to {jsonl_path}")
    except Exception as e:
        print(f"  [longbench] FAILED: {e}")
        meta["error"] = str(e)
    return meta


def _download_zeroscrolls(force: bool = False) -> dict:
    """Download tau/zero_scrolls — raw zip files from HF Hub."""
    from huggingface_hub import hf_hub_download

    target = EVAL_DIR / "zeroscrolls"
    meta = {"name": "ZeroSCROLLS", "source": "tau/zero_scrolls"}

    tasks = [
        "gov_report", "summ_screen_fd", "qmsum", "squality",
        "qasper", "narrative_qa", "quality", "musique",
        "space_digest", "book_sum_sort",
    ]

    total_rows = 0
    task_info = {}
    for task in tasks:
        task_dir = target / task
        jsonl_candidates = list(task_dir.glob("*.jsonl")) if task_dir.exists() else []
        if jsonl_candidates and not force:
            n = sum(1 for _ in open(jsonl_candidates[0]))
            task_info[task] = n
            total_rows += n
            print(f"  [zeroscrolls] {task} already cached: {n} examples")
            continue

        try:
            print(f"  [zeroscrolls] Downloading {task}.zip...")
            zip_path = hf_hub_download(
                repo_id="tau/zero_scrolls",
                filename=f"{task}.zip",
                repo_type="dataset",
                local_dir=str(target),
            )
            # Extract
            task_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(task_dir))

            # Count rows in validation split
            val_files = list(task_dir.rglob("*val*")) + list(task_dir.rglob("*validation*"))
            jsonl_files = [f for f in task_dir.rglob("*.jsonl")]
            data_file = val_files[0] if val_files else (jsonl_files[0] if jsonl_files else None)

            if data_file and data_file.exists():
                n = sum(1 for _ in open(data_file))
                task_info[task] = n
                total_rows += n
                print(f"  [zeroscrolls] {task}: {n} examples ({data_file.name})")
            else:
                # List what we extracted
                extracted = list(task_dir.rglob("*"))
                task_info[task] = f"extracted {len(extracted)} files"
                print(f"  [zeroscrolls] {task}: extracted {len(extracted)} files")

        except Exception as e:
            print(f"  [zeroscrolls] {task} FAILED: {e}")
            task_info[task] = f"ERROR: {e}"

    meta["tasks"] = task_info
    meta["total_rows"] = total_rows
    meta["note"] = "Raw zip downloads from HF Hub (datasets v4 compat)"
    return meta


def _download_leval(force: bool = False) -> dict:
    """Download L4NLP/LEval — raw files from HF Hub."""
    from huggingface_hub import HfApi, hf_hub_download

    target = EVAL_DIR / "leval"
    meta = {"name": "L-Eval", "source": "L4NLP/LEval", "license": "CC-BY-4.0"}

    api = HfApi()
    try:
        files = api.list_repo_files("L4NLP/LEval", repo_type="dataset")
    except Exception as e:
        print(f"  [leval] Could not list files: {e}")
        meta["error"] = str(e)
        return meta

    # Download all data files (jsonl, json, zip)
    data_files = [f for f in files if f.endswith((".jsonl", ".json", ".zip")) and not f.startswith(".")]
    print(f"  [leval] Found {len(data_files)} data files")

    total_rows = 0
    config_info = {}
    for fname in data_files:
        local_path = target / fname
        if local_path.exists() and not force:
            if fname.endswith(".jsonl"):
                n = sum(1 for _ in open(local_path))
                config_info[fname] = n
                total_rows += n
            print(f"  [leval] {fname} already cached")
            continue

        try:
            print(f"  [leval] Downloading {fname}...")
            hf_hub_download(
                repo_id="L4NLP/LEval",
                filename=fname,
                repo_type="dataset",
                local_dir=str(target),
            )

            if fname.endswith(".jsonl"):
                n = sum(1 for _ in open(target / fname))
                config_info[fname] = n
                total_rows += n
                print(f"  [leval] {fname}: {n} rows")
            elif fname.endswith(".zip"):
                with zipfile.ZipFile(str(target / fname), "r") as zf:
                    zf.extractall(str(target))
                config_info[fname] = "extracted"
                print(f"  [leval] {fname}: extracted")
            else:
                config_info[fname] = "downloaded"

        except Exception as e:
            print(f"  [leval] {fname} FAILED: {e}")
            config_info[fname] = f"ERROR: {e}"

    meta["files"] = config_info
    meta["total_rows"] = total_rows
    return meta


def _compute_disk_sizes() -> dict:
    """Compute disk usage per dataset directory."""
    sizes = {}
    for subdir in EVAL_DIR.iterdir():
        if subdir.is_dir():
            total = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
            sizes[subdir.name] = f"{total / (1024**2):.1f} MB"
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Download long-context eval datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    print("=" * 60)
    print("Long-Context Evaluation Dataset Download")
    print(f"Target: {EVAL_DIR}")
    print("=" * 60)

    metadata = {
        "download_timestamp": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }

    t0 = time.time()

    print("\n--- LongBench v2 ---")
    metadata["datasets"]["longbench"] = _download_longbench_v2(args.force)

    print("\n--- ZeroSCROLLS ---")
    metadata["datasets"]["zeroscrolls"] = _download_zeroscrolls(args.force)

    print("\n--- L-Eval ---")
    metadata["datasets"]["leval"] = _download_leval(args.force)

    # Check git repos
    print("\n--- RULER (git repo) ---")
    ruler_ok = (EVAL_DIR / "ruler" / "repo" / "scripts").exists()
    metadata["datasets"]["ruler"] = {
        "name": "RULER", "source": "github.com/hsiehjackson/RULER",
        "license": "Apache-2.0", "status": "cloned" if ruler_ok else "MISSING",
    }
    print(f"  Status: {'cloned' if ruler_ok else 'MISSING'}")

    print("\n--- Needle-in-a-Haystack (git repo) ---")
    needle_ok = (EVAL_DIR / "needle" / "repo" / "needlehaystack").exists()
    metadata["datasets"]["needle"] = {
        "name": "Needle-in-a-Haystack", "source": "github.com/gkamradt/LLMTest_NeedleInAHaystack",
        "license": "MIT", "status": "cloned" if needle_ok else "MISSING",
    }
    print(f"  Status: {'cloned' if needle_ok else 'MISSING'}")

    # Disk sizes
    print("\n--- Disk Usage ---")
    metadata["disk_usage"] = _compute_disk_sizes()
    for name, size in sorted(metadata["disk_usage"].items()):
        print(f"  {name}: {size}")

    elapsed = time.time() - t0
    metadata["download_duration_seconds"] = round(elapsed, 1)
    print(f"\nTotal download time: {elapsed:.1f}s")

    meta_path = EVAL_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata written to {meta_path}")


if __name__ == "__main__":
    main()
