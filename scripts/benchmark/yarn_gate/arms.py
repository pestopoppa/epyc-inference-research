"""YARN-AGG-1: the aggregation / recall arms of the YaRN gate as DATA (spec = this file + the handoff section).

Comparisons: A0 vs A1 (GPU 27B) and CN vs CY2 (CPU 35B-A3B hybrid), at native lengths and beyond. Nothing
here runs inference; ``build_manifest()`` lists the cells a window runner must execute, each carrying its
pins. Zone labels are NEVER taken from these nominal lengths (see ``retokenize.label_zone``).
"""

from __future__ import annotations

from typing import Any

PAIRS = (("A0", "A1"), ("CN", "CY2"))
NOMINAL_LENGTHS_NATIVE = (131072, 245760)  # 128K, 240K
NOMINAL_LENGTHS_BEYOND = (409600, 512000)  # 400K, 500K

#: Pre-declared before any run (2026-10-21, YARN-AGG-1). Operator may amend ONLY before the first run.
MARGINS = {
    # Operator-ratified "Net, paired" form (2026-10-21). No absolute slack.
    "in_native_rule": "Net, paired: inside the native window A1 FAILS iff a one-sided exact McNemar test on the "
                      "paired discordant items (A0 right / A1 wrong = b, versus A0 wrong / A1 right = c) is "
                      "significant at alpha = 0.05, i.e. P(X >= b | n = b + c, p = 0.5) <= 0.05; otherwise NOT-FAIL. "
                      "The point estimate (mean paired score difference, b, c) is always reported beside the verdict.",
    "alpha": 0.05,
    "right_threshold": "an item is RIGHT iff its score >= 1.0 (fully correct); partial-credit scores count as wrong",
    "needle_cells_keep": "A1 >= A0 - 1 correct of 10 (E1 rule, unchanged)",
    "bins": "REAL prompt_n, step 32768; zone from retokenize.label_zone only",
}

OOLONG = {
    "arm": "oolong",
    "adapter": "scripts/benchmark/oolong_adapter.py (OOL-1, lane dpf-sample-yarn-ool); this arm is config + aggregation on top",
    "dataset": "oolongbench/oolong-synth", "revision": "81d2f8d9349f6ff329d695f9be494051e9f05cb9",
    "upstream_scorer": "abertsch72/oolong@0bb7eabe8 (vendored unmodified)",
    "tasks": ["counting", "user"],
    # Targets are REAL-token zones (128K/240K native, 400K/500K beyond). The dataset ships power-of-two NOMINAL
    # context_len buckets (13 lengths, OL3/OL6), so 240K/400K/500K probably do not exist as buckets: the candidate
    # buckets below are checked by fetch_oolong_subset.py --list-lengths and re-measured with the served tokenizer.
    # A target zone with no bucket landing in it is reported NOT CONSTRUCTIBLE, never relabelled from nominal.
    "target_real_lengths": [131072, 245760, 409600, 512000],
    "nominal_lengths": [131072, 262144, 524288],
    "split": "test", "labels_given": False,
    "per_bin_outputs": ["mean score (+ context-clustered CI)", "attempted_rate", "no_answer_rate", "failure classes"],
    "control": {
        "name": "HoloBench info_amount (intake-2268#record): constant relevant amount",
        "method": "restrict to items whose num_labels lies in the band common to ALL lengths; report score by "
                  "real-length bin inside the band, and score by num_labels band inside each length. A drop at "
                  "400K counts as LENGTH only if it persists inside the matched band; if the band is empty at a "
                  "length the control is reported UNAVAILABLE and the drop is not attributed.",
    },
    "risk": "oolong-synth test has 13 nominal lengths x 400 items, about 16 distinct contexts per length "
            "(OL6): CIs must be clustered by context; verify 240K/400K/500K nominal buckets exist before the fetch.",
}

HELMET = {
    "arm": "helmet", "repo": "https://github.com/princeton-nlp/HELMET",
    "pin": "aeadacc6593e9a222a84b0660b186a4e23d2efda",
    "endpoint_url": "http://localhost:8080/v1/",
    "cells": {
        "recall": ["ruler_niah_mk_2", "ruler_niah_mk_3", "ruler_niah_mv", "json_kv"],
        "rag": ["nq", "hotpotqa"],
    },
    "nominal_lengths": [131072, 245760],
    "record": ["config file path + sha256", "max_test_samples", "pin", "real prompt_n per row"],
    "comparability": "paper numbers NON-COMPARABLE (different model, templates, tokenizer-sized lengths)",
    "judge": "none for recall/rag (substring/exact-match metrics); judged categories go through YARN-JUDGE-1",
    "unverified": "HELMET ships configs up to 128K; the 240K config is a local copy recorded by sha and "
                  "must be validated by the window runner against the pinned checkout before use",
}

LONGBIO = {
    "arm": "longbio", "generator": "LongBioBench generator",
    "generator_pin": "dc6f6ed1aab5dd072f0837f3160a45beb4df8dbe",
    "variants": {"standard": {"retrievals": 1}, "multi_standard_5": {"retrievals": 5},
                 "multi_standard_10": {"retrievals": 10}, "paraphrase": {"retrievals": 1}},
    "tokenizer": "served-model tokenizer (retokenize.ServedTokenizer); the generator sizes lengths by its own tokenizer",
    "grader": "exact match (normalised: lowercase, strip, collapse whitespace)",
    "haystack": "coherent (generator filler text), NOT the neutral RULER-noise haystack (intake-2309#record)",
    "nominal_lengths": [131072, 245760, 409600, 512000],
    "unverified": "generator CLI/output schema not inspected offline; runner reads JSONL rows "
                  "{id, prompt|context, question, answer, variant} and the window runner must adapt the generator output to it",
}

ARMS = {"oolong": OOLONG, "helmet": HELMET, "longbio": LONGBIO}


def helmet_command(task_group: str, task: str, *, helmet_dir: str, config_file: str, max_test_samples: int,
                   model_name: str, output_dir: str) -> list[str]:
    """Argv for the pinned HELMET ``eval.py`` against the local server. Built, never executed here."""
    assert task_group in HELMET["cells"] and task in HELMET["cells"][task_group], (task_group, task)
    return ["python3", f"{helmet_dir}/eval.py", "--config", config_file,
            "--endpoint_url", HELMET["endpoint_url"], "--model_name_or_path", model_name,
            "--max_test_samples", str(max_test_samples), "--output_dir", output_dir]


def build_manifest(role_arms: tuple[str, ...] = ("A0", "A1", "CN", "CY2")) -> list[dict[str, Any]]:
    cells = []
    for arm_name, spec in ARMS.items():
        for role in role_arms:
            for nom in spec["nominal_lengths"]:
                if arm_name == "oolong":
                    for t in spec["tasks"]:
                        cells.append({"arm": arm_name, "role": role, "nominal_length": nom, "task": t})
                elif arm_name == "helmet":
                    for g, ts in spec["cells"].items():
                        for t in ts:
                            cells.append({"arm": arm_name, "role": role, "nominal_length": nom, "group": g, "task": t})
                else:
                    for v in spec["variants"]:
                        cells.append({"arm": arm_name, "role": role, "nominal_length": nom, "task": v})
    for c in cells:
        c["zone"] = "UNLABELLED (set from real prompt_n after the run)"
    return cells
