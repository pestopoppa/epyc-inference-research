#!/usr/bin/env python3
from __future__ import annotations

"""Reconstruct benchmark suites from HuggingFace datasets.

Uses MANIFEST.yaml to map suites to their source datasets and
regenerates prompt YAML files from the original data.

Usage:
    python reconstruct_suite.py --suite math --output benchmarks/prompts/v2/math.yaml
    python reconstruct_suite.py --suite all --output-dir benchmarks/prompts/v2/
    python reconstruct_suite.py --suite math --dry-run
    python reconstruct_suite.py --list
"""

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

MANIFEST_PATH = PROJECT_ROOT / "benchmarks" / "prompts" / "MANIFEST.yaml"


def load_manifest() -> dict:
    """Load the benchmark manifest."""
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def list_suites(manifest: dict) -> None:
    """List all available suites."""
    print("Available benchmark suites:\n")
    print("HuggingFace-backed suites:")
    print("-" * 60)
    for name, info in manifest.get("suites", {}).items():
        sources = ", ".join(s["name"] for s in info.get("sources", []))
        print(f"  {name:25} ({info.get('count', '?'):>5} questions)")
        print(f"    Sources: {sources}")
        print()

    print("\nYAML-only suites (no reconstruction):")
    print("-" * 60)
    for name, info in manifest.get("yaml_only_suites", {}).items():
        print(f"  {name:25} - {info.get('reason', 'N/A')}")


def reconstruct_suite(
    suite_name: str,
    manifest: dict,
    output_path: Path | None = None,
    sample_size: int = 100,
    seed: int = 42,
    dry_run: bool = False,
) -> bool:
    """Reconstruct a single suite from HuggingFace.

    Args:
        suite_name: Name of the suite to reconstruct.
        manifest: Loaded manifest dict.
        output_path: Path to write output YAML.
        sample_size: Number of questions to sample.
        seed: Random seed for sampling.
        dry_run: If True, don't write files.

    Returns:
        True if successful, False otherwise.
    """
    suites = manifest.get("suites", {})

    if suite_name not in suites:
        yaml_only = manifest.get("yaml_only_suites", {})
        if suite_name in yaml_only:
            print(f"Suite '{suite_name}' is YAML-only (no HuggingFace source)")
            print(f"  Source: {yaml_only[suite_name].get('source', 'N/A')}")
            print(f"  Reason: {yaml_only[suite_name].get('reason', 'N/A')}")
            return False
        print(f"ERROR: Unknown suite '{suite_name}'")
        return False

    suite_info = suites[suite_name]
    adapter_name = suite_info.get("adapter")

    print(f"Reconstructing suite: {suite_name}")
    print(f"  Adapter: {adapter_name}")
    print(f"  Sources: {', '.join(s['name'] for s in suite_info.get('sources', []))}")
    print(f"  Sample size: {sample_size}")
    print(f"  Seed: {seed}")

    if dry_run:
        print("\n[DRY RUN] Would execute:")
        print(f"  {suite_info.get('reconstruction_command', 'N/A').strip()}")
        return True

    # Import and use the adapter
    try:
        from dataset_adapters import get_adapter
    except ImportError:
        print("ERROR: Could not import dataset_adapters module")
        print("  Make sure you're running from the scripts/benchmark directory")
        return False

    adapter = get_adapter(suite_name)
    if adapter is None:
        print(f"ERROR: No adapter found for suite '{suite_name}'")
        return False

    print(f"\nLoading dataset...")
    try:
        questions = adapter.sample(n=sample_size, seed=seed)
    except Exception as e:
        print(f"ERROR: Failed to sample from adapter: {e}")
        return False

    print(f"  Sampled {len(questions)} questions")

    if not questions:
        print("ERROR: No questions sampled")
        return False

    # Convert to YAML format
    yaml_content = {
        "suite": suite_name,
        "version": "reconstructed",
        "source": "HuggingFace",
        "seed": seed,
        "questions": [],
    }

    for q in questions:
        yaml_q = {
            "id": q.get("id", "unknown"),
            "tier": q.get("tier", 1),
            "prompt": q.get("prompt", ""),
            "expected": q.get("expected", ""),
            "scoring_method": q.get("scoring_method", "exact_match"),
        }
        if q.get("context"):
            yaml_q["context"] = q["context"]
        if q.get("image_path"):
            yaml_q["image_path"] = q["image_path"]
        if q.get("scoring_config"):
            yaml_q["scoring_config"] = q["scoring_config"]
        yaml_content["questions"].append(yaml_q)

    # Determine output path
    if output_path is None:
        output_path = PROJECT_ROOT / "benchmarks" / "prompts" / "reconstructed" / f"{suite_name}.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nWritten to: {output_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct benchmark suites from HuggingFace datasets"
    )
    parser.add_argument(
        "--suite",
        type=str,
        help="Suite name to reconstruct (or 'all' for all suites)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output YAML file path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for multiple suites",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=100,
        help="Number of questions to sample (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available suites",
    )

    args = parser.parse_args()

    manifest = load_manifest()

    if args.list:
        list_suites(manifest)
        return 0

    if not args.suite:
        parser.print_help()
        return 1

    if args.suite == "all":
        # Reconstruct all HuggingFace-backed suites
        output_dir = args.output_dir or (PROJECT_ROOT / "benchmarks" / "prompts" / "reconstructed")
        success_count = 0
        fail_count = 0

        for suite_name in manifest.get("suites", {}):
            output_path = output_dir / f"{suite_name}.yaml"
            success = reconstruct_suite(
                suite_name,
                manifest,
                output_path=output_path,
                sample_size=args.sample,
                seed=args.seed,
                dry_run=args.dry_run,
            )
            if success:
                success_count += 1
            else:
                fail_count += 1
            print()

        print(f"\nSummary: {success_count} succeeded, {fail_count} failed")
        return 0 if fail_count == 0 else 1

    # Single suite
    success = reconstruct_suite(
        args.suite,
        manifest,
        output_path=args.output,
        sample_size=args.sample,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
