#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from suites import get_all_suite_names, load_suite


def test_load_suite_supports_mapping_prompts(tmp_path):
    (tmp_path / "mapping.yaml").write_text(
        yaml.safe_dump(
            {
                "description": "mapping suite",
                "prompts": {
                    "q1": {
                        "prompt": "What is 2+2?",
                        "expected": "4",
                        "tier": 2,
                    }
                },
            }
        )
    )

    suite = load_suite("mapping", prompts_dir=str(tmp_path))

    assert suite is not None
    assert [q.id for q in suite.questions] == ["q1"]
    assert suite.questions[0].prompt == "What is 2+2?"
    assert suite.questions[0].expected == "4"
    assert suite.questions[0].tier == 2


def test_load_suite_supports_list_prompts(tmp_path):
    (tmp_path / "list_gate.yaml").write_text(
        yaml.safe_dump(
            {
                "description": "list suite",
                "prompts": [
                    {
                        "id": "factual_01",
                        "category": "short_factual",
                        "text": "Who wrote Hamlet?",
                    }
                ],
            }
        )
    )

    suite = load_suite("list_gate", prompts_dir=str(tmp_path))

    assert suite is not None
    assert [q.id for q in suite.questions] == ["factual_01"]
    assert suite.questions[0].prompt == "Who wrote Hamlet?"
    assert suite.questions[0].name == "factual_01"
    assert suite.questions[0].expected == ""


def test_get_all_suite_names_preserves_yaml_only_default(tmp_path):
    (tmp_path / "yaml_only.yaml").write_text("prompts: {}\n")

    names = get_all_suite_names(prompts_dir=str(tmp_path))

    assert names == ["yaml_only"]


def test_get_all_suite_names_can_include_adapter_suites(tmp_path):
    (tmp_path / "yaml_only.yaml").write_text("prompts: {}\n")

    names = get_all_suite_names(prompts_dir=str(tmp_path), include_adapters=True)

    assert "yaml_only" in names
    assert "omniscience" in names
    assert "tulving_episodic" in names
