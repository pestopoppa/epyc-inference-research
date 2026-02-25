#!/usr/bin/env python3
from __future__ import annotations

"""Extract real VL benchmark questions from HuggingFace-cached Arrow files.

Reads OCRBench and ChartQA Arrow datasets, samples diverse questions,
extracts images to disk, and generates benchmarks/prompts/debug/vl.yaml
with real benchmark Q&A pairs and proper scoring methods.

Also provides an adapter interface for on-the-fly sampling in the
learning loop (import VLDatasetAdapter and call .sample()).

Usage:
    # Generate static debug suite YAML (42+ questions)
    python scripts/benchmark/extract_vl_debug_suite.py

    # Generate with custom count
    python scripts/benchmark/extract_vl_debug_suite.py --total 60

    # Dry-run: print what would be generated without writing
    python scripts/benchmark/extract_vl_debug_suite.py --dry-run
"""

import argparse
import io
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import datasets as hf_datasets
except ImportError:
    print("ERROR: datasets not installed. Run: pip install datasets", file=sys.stderr)
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────

IMAGE_DIR = PROJECT_ROOT / "benchmarks" / "images" / "vl"
YAML_OUT = PROJECT_ROOT / "benchmarks" / "prompts" / "debug" / "vl.yaml"

# HuggingFace dataset identifiers (loaded from local cache)
HF_DATASET_IDS = {
    "ocrbench": "echo840/OCRBench",
    "chartqa": "HuggingFaceM4/ChartQA",
}

# OCRBench question type → difficulty tier mapping
OCRBENCH_TIER_MAP = {
    "Regular Text Recognition": 1,
    "Irregular Text Recognition": 1,
    "Artistic Text Recognition": 1,
    "Digit String Recognition": 1,
    "Handwriting Recognition": 2,
    "Handwritten Math Expression Recognition": 3,
    "Non-Semantic Text Recognition": 2,
    "Scene Text-centric VQA": 2,
    "Doc-oriented VQA": 3,
    "Key Information Extraction": 2,
}

# How to score each OCRBench question type
OCRBENCH_SCORING = {
    "Regular Text Recognition": ("exact_match", {"case_sensitive": False}),
    "Irregular Text Recognition": ("exact_match", {"case_sensitive": False}),
    "Artistic Text Recognition": ("exact_match", {"case_sensitive": False}),
    "Digit String Recognition": ("exact_match", {}),
    "Handwriting Recognition": ("exact_match", {"case_sensitive": False}),
    "Handwritten Math Expression Recognition": ("substring", {"case_sensitive": False}),
    "Non-Semantic Text Recognition": ("exact_match", {"case_sensitive": False}),
    "Scene Text-centric VQA": ("substring", {"case_sensitive": False}),
    "Doc-oriented VQA": ("substring", {"case_sensitive": False}),
    "Key Information Extraction": ("substring", {"case_sensitive": False}),
}


# ── Data Classes ─────────────────────────────────────────────────────────


@dataclass
class VLQuestion:
    """A single VL benchmark question with image."""
    id: str
    source: str               # "ocrbench" or "chartqa"
    source_index: int          # Row index in original dataset
    question: str
    answers: list[str]         # Acceptable answers (list)
    image_bytes: bytes
    tier: int                  # 1=easy, 2=medium, 3=hard
    scoring_method: str        # "exact_match" or "substring"
    scoring_config: dict = field(default_factory=dict)
    question_type: str = ""    # OCRBench question_type or "chart_qa"
    image_path: str = ""       # Set after extraction


# ── Dataset Loading (via HuggingFace datasets library) ───────────────────


def _image_to_bytes(pil_image) -> bytes:
    """Convert a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def load_ocrbench() -> list[VLQuestion]:
    """Load all OCRBench questions from HuggingFace cache."""
    try:
        ds = hf_datasets.load_dataset(HF_DATASET_IDS["ocrbench"], split="test")
    except Exception as e:
        print(f"WARNING: Could not load OCRBench: {e}", file=sys.stderr)
        return []

    questions = []
    for idx, row in enumerate(ds):
        qt = row.get("question_type", "")
        scoring_method, scoring_config = OCRBENCH_SCORING.get(
            qt, ("substring", {"case_sensitive": False})
        )

        img = row.get("image")
        img_bytes = _image_to_bytes(img) if img else b""

        answers = row.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]

        questions.append(VLQuestion(
            id=f"ocr_{idx:04d}",
            source="ocrbench",
            source_index=idx,
            question=row.get("question", ""),
            answers=answers,
            image_bytes=img_bytes,
            tier=OCRBENCH_TIER_MAP.get(qt, 2),
            scoring_method=scoring_method,
            scoring_config=scoring_config,
            question_type=qt,
        ))

    return questions


def load_chartqa(split: str = "test") -> list[VLQuestion]:
    """Load ChartQA questions from HuggingFace cache."""
    try:
        ds = hf_datasets.load_dataset(HF_DATASET_IDS["chartqa"], split=split)
    except Exception as e:
        print(f"WARNING: Could not load ChartQA ({split}): {e}", file=sys.stderr)
        return []

    questions = []
    for idx, row in enumerate(ds):
        img = row.get("image")
        img_bytes = _image_to_bytes(img) if img else b""

        labels = row.get("label", [])
        if isinstance(labels, str):
            labels = [labels]

        is_human = row.get("human_or_machine", 0) == 0
        tier = 2 if is_human else 1

        questions.append(VLQuestion(
            id=f"chart_{split}_{idx:04d}",
            source="chartqa",
            source_index=idx,
            question=row.get("query", ""),
            answers=labels,
            image_bytes=img_bytes,
            tier=tier,
            scoring_method="substring",
            scoring_config={"case_sensitive": False},
            question_type=f"chartqa_{'human' if is_human else 'machine'}",
        ))

    return questions


# ── Sampling ─────────────────────────────────────────────────────────────


def sample_diverse(
    questions: list[VLQuestion],
    n: int,
    seed: int = 42,
    group_key: str = "question_type",
) -> list[VLQuestion]:
    """Sample n questions with diversity across group_key.

    Ensures proportional representation of each group, then fills
    remaining slots randomly.
    """
    rng = random.Random(seed)

    # Group by key
    groups: dict[str, list[VLQuestion]] = {}
    for q in questions:
        key = getattr(q, group_key, "unknown")
        groups.setdefault(key, []).append(q)

    # Proportional allocation
    sampled = []
    per_group = max(1, n // len(groups))
    remainder = n - per_group * len(groups)

    for key in sorted(groups.keys()):
        pool = groups[key]
        take = min(per_group, len(pool))
        sampled.extend(rng.sample(pool, take))

    # Fill remainder from all remaining
    already = {id(q) for q in sampled}
    remaining = [q for q in questions if id(q) not in already]
    if remainder > 0 and remaining:
        sampled.extend(rng.sample(remaining, min(remainder, len(remaining))))

    # If still short, take more from largest groups
    while len(sampled) < n:
        already = {id(q) for q in sampled}
        remaining = [q for q in questions if id(q) not in already]
        if not remaining:
            break
        sampled.append(rng.choice(remaining))

    return sampled[:n]


def assign_tiers(questions: list[VLQuestion]) -> list[VLQuestion]:
    """Ensure balanced tier distribution across the final set.

    Re-assigns tiers to ensure roughly:
    - Tier 1: 40% (easy — basic OCR, simple chart counting)
    - Tier 2: 35% (medium — VQA, extraction, chart reading)
    - Tier 3: 25% (hard — document reasoning, math expressions)
    """
    n = len(questions)
    t1_target = int(n * 0.40)
    t2_target = int(n * 0.35)

    # Sort by original tier to get natural ordering
    by_tier = sorted(questions, key=lambda q: (q.tier, q.question_type))

    for i, q in enumerate(by_tier):
        if i < t1_target:
            q.tier = 1
        elif i < t1_target + t2_target:
            q.tier = 2
        else:
            q.tier = 3

    return by_tier


# ── Image Extraction ─────────────────────────────────────────────────────


def extract_image(q: VLQuestion, base_dir: Path) -> str:
    """Save image bytes to disk, return the absolute path."""
    if not q.image_bytes:
        return ""

    # Determine subdirectory
    if q.source == "ocrbench":
        subdir = "ocrbench"
    elif q.source == "chartqa":
        subdir = "chartqa"
    else:
        subdir = "other"

    out_dir = base_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{q.id}.png"
    out_path = out_dir / filename

    # Detect format from bytes
    if q.image_bytes[:4] == b'\x89PNG':
        ext = ".png"
    elif q.image_bytes[:2] == b'\xff\xd8':
        ext = ".jpg"
        filename = f"{q.id}.jpg"
        out_path = out_dir / filename
    elif q.image_bytes[:4] == b'RIFF':
        ext = ".webp"
        filename = f"{q.id}.webp"
        out_path = out_dir / filename
    else:
        # Try to decode with PIL to determine format
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(q.image_bytes))
            fmt = img.format or "PNG"
            ext = f".{fmt.lower()}"
            filename = f"{q.id}{ext}"
            out_path = out_dir / filename
            img.save(str(out_path))
            return str(out_path)
        except Exception as e:
            ext = ".png"

    with open(out_path, "wb") as f:
        f.write(q.image_bytes)

    return str(out_path)


# ── YAML Generation ──────────────────────────────────────────────────────


def _yaml_escape(s: str) -> str:
    """Escape a string for YAML double-quoted context."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def generate_yaml(questions: list[VLQuestion]) -> str:
    """Generate the vl.yaml debug suite from extracted questions."""
    lines = [
        "# Debug suite: Vision-Language (real benchmark images)",
        "# Sources: OCRBench (echo840/OCRBench), ChartQA (HuggingFaceM4/ChartQA)",
        "# All questions and answers are extracted verbatim from published datasets.",
        "# Images extracted from HuggingFace Arrow cache to benchmarks/images/vl/",
        "#",
        "# Generated by: scripts/benchmark/extract_vl_debug_suite.py",
        "# Scoring: exact_match, substring",
        "",
        "suite: vl",
        'version: "3.0"',
        "scoring_default:",
        "  method: substring",
        "",
        "questions:",
    ]

    # Group by source for readability
    ocr_qs = [q for q in questions if q.source == "ocrbench"]
    chart_qs = [q for q in questions if q.source == "chartqa"]

    if ocr_qs:
        lines.append("")
        lines.append("  # ── OCRBench (echo840/OCRBench) ─────────────────────────────────────")
        lines.append("")
        for q in sorted(ocr_qs, key=lambda x: (x.tier, x.id)):
            lines.extend(_question_to_yaml(q))

    if chart_qs:
        lines.append("")
        lines.append("  # ── ChartQA (HuggingFaceM4/ChartQA) ────────────────────────────────")
        lines.append("")
        for q in sorted(chart_qs, key=lambda x: (x.tier, x.id)):
            lines.extend(_question_to_yaml(q))

    lines.append("")
    return "\n".join(lines)


def _question_to_yaml(q: VLQuestion) -> list[str]:
    """Convert a VLQuestion to YAML lines."""
    lines = []
    lines.append(f"  - id: vl_{q.id}")
    lines.append(f"    tier: {q.tier}")
    lines.append(f'    image_path: "{q.image_path}"')
    lines.append(f"    source_dataset: {q.source}")
    lines.append(f"    source_index: {q.source_index}")

    # Prompt — use the original question verbatim
    prompt = q.question.strip()
    if "\n" in prompt:
        lines.append("    prompt: |")
        for pline in prompt.split("\n"):
            lines.append(f"      {pline}")
    else:
        lines.append(f'    prompt: "{_yaml_escape(prompt)}"')

    # Expected answer — use first answer from the list
    expected = q.answers[0] if q.answers else ""
    lines.append(f'    expected: "{_yaml_escape(expected)}"')

    # Additional acceptable answers
    if len(q.answers) > 1:
        lines.append("    alt_answers:")
        for alt in q.answers[1:]:
            lines.append(f'      - "{_yaml_escape(alt)}"')

    lines.append(f"    scoring_method: {q.scoring_method}")
    if q.scoring_config:
        lines.append("    scoring_config:")
        for k, v in q.scoring_config.items():
            if isinstance(v, bool):
                lines.append(f"      {k}: {'true' if v else 'false'}")
            else:
                lines.append(f'      {k}: "{v}"')

    lines.append("")
    return lines


# ── VL Dataset Adapter (for on-the-fly sampling) ────────────────────────


class VLDatasetAdapter:
    """Adapter for on-the-fly sampling from VL benchmark datasets.

    Usage in learning loop:
        adapter = VLDatasetAdapter()
        questions = adapter.sample(n=10, seed=epoch_num)
        for q in questions:
            # q has: id, prompt, expected, image_path, scoring_method, etc.
            pass
    """

    def __init__(self, image_dir: Optional[Path] = None):
        self.image_dir = image_dir or IMAGE_DIR
        self._ocrbench: Optional[list[VLQuestion]] = None
        self._chartqa: Optional[list[VLQuestion]] = None

    def _ensure_loaded(self):
        if self._ocrbench is None:
            self._ocrbench = load_ocrbench()
        if self._chartqa is None:
            self._chartqa = load_chartqa("test")

    @property
    def total_available(self) -> int:
        self._ensure_loaded()
        return len(self._ocrbench) + len(self._chartqa)

    def sample(
        self,
        n: int = 42,
        seed: int = 42,
        extract_images: bool = True,
    ) -> list[dict]:
        """Sample n questions, extract images, return prompt dicts.

        Returns dicts compatible with compare_orchestrator_direct.py format:
        {id, suite, prompt, expected, image_path, scoring_method, ...}
        """
        self._ensure_loaded()
        all_qs = self._ocrbench + self._chartqa

        # Split sampling: ~55% OCRBench, ~45% ChartQA
        n_ocr = int(n * 0.55)
        n_chart = n - n_ocr

        sampled = []
        sampled.extend(sample_diverse(self._ocrbench, n_ocr, seed=seed))
        sampled.extend(sample_diverse(self._chartqa, n_chart, seed=seed))
        sampled = assign_tiers(sampled)

        if extract_images:
            for q in sampled:
                q.image_path = extract_image(q, self.image_dir)

        return [self._to_prompt_dict(q) for q in sampled]

    @staticmethod
    def _to_prompt_dict(q: VLQuestion) -> dict:
        return {
            "id": f"vl_{q.id}",
            "suite": "vl",
            "prompt": q.question.strip(),
            "context": "",
            "expected": q.answers[0] if q.answers else "",
            "scoring": [],
            "image_path": q.image_path,
            "tier": q.tier,
            "scoring_method": q.scoring_method,
            "scoring_config": q.scoring_config,
        }


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract VL debug suite from real benchmark datasets"
    )
    parser.add_argument(
        "--total", type=int, default=42,
        help="Total questions to include (default: 42)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing files"
    )
    parser.add_argument(
        "--output", type=str, default=str(YAML_OUT),
        help="Output YAML path"
    )
    args = parser.parse_args()

    print("Loading OCRBench...")
    ocr_qs = load_ocrbench()
    print(f"  Loaded {len(ocr_qs)} questions across "
          f"{len(set(q.question_type for q in ocr_qs))} types")

    print("Loading ChartQA (test split)...")
    chart_qs = load_chartqa("test")
    print(f"  Loaded {len(chart_qs)} questions "
          f"({sum(1 for q in chart_qs if 'human' in q.question_type)} human, "
          f"{sum(1 for q in chart_qs if 'machine' in q.question_type)} machine)")

    # Sample with diversity
    n_ocr = int(args.total * 0.55)
    n_chart = args.total - n_ocr
    print(f"\nSampling {n_ocr} OCRBench + {n_chart} ChartQA = {args.total} total")

    sampled = []
    sampled.extend(sample_diverse(ocr_qs, n_ocr, seed=args.seed))
    sampled.extend(sample_diverse(chart_qs, n_chart, seed=args.seed))
    sampled = assign_tiers(sampled)

    # Print distribution
    tier_counts = {}
    type_counts = {}
    for q in sampled:
        tier_counts[q.tier] = tier_counts.get(q.tier, 0) + 1
        type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1

    print(f"\nTier distribution:")
    for t in sorted(tier_counts):
        print(f"  Tier {t}: {tier_counts[t]}")

    print(f"\nQuestion type distribution:")
    for qt in sorted(type_counts):
        print(f"  {qt}: {type_counts[qt]}")

    if args.dry_run:
        print("\n[DRY RUN] Would generate YAML and extract images. Exiting.")
        return

    # Extract images
    print(f"\nExtracting images to {IMAGE_DIR}/...")
    extracted = 0
    failed = 0
    for q in sampled:
        path = extract_image(q, IMAGE_DIR)
        if path:
            q.image_path = path
            extracted += 1
        else:
            failed += 1
            print(f"  WARNING: No image for {q.id}")

    print(f"  Extracted {extracted} images ({failed} failed)")

    # Generate YAML
    yaml_content = generate_yaml(sampled)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(yaml_content)
    print(f"\nWrote {out_path} ({len(sampled)} questions)")

    # Summary
    print(f"\n{'='*60}")
    print(f"VL Debug Suite Generated")
    print(f"  Questions: {len(sampled)}")
    print(f"  Sources: OCRBench ({n_ocr}), ChartQA ({n_chart})")
    print(f"  Images: {IMAGE_DIR}/")
    print(f"  YAML: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
