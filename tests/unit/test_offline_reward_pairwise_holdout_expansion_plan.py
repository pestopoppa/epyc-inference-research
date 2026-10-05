from __future__ import annotations

import json
from pathlib import Path

from scripts.graph_router import plan_offline_reward_pairwise_holdout_expansion as mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _manifest_row(
    *,
    source_path: str,
    role_key: str,
    expected_sha256: str = "expected-hash",
    prompt_sha256: str = "prompt-hash",
) -> dict:
    return {
        "schema_version": "offline_reward_feature_input.v1",
        "item_id": f"{Path(source_path).stem}:1:{role_key}",
        "question_id": "q-live",
        "suite": "livecodebench",
        "role_key": role_key,
        "source_path": source_path,
        "source_record_offset": 0,
        "prompt_sha256": prompt_sha256,
        "expected_sha256": expected_sha256,
        "feature_context": {"source_family": "seeding_eval"},
    }


def _result_record() -> dict:
    return {
        "question_id": "q-live",
        "suite": "livecodebench",
        "prompt": "Solve task",
        "expected": "42",
        "role_results": {
            "frontdoor:direct": {
                "answer": "42",
                "passed": True,
                "elapsed_seconds": 1.0,
            },
            "coder_primary": {
                "answer": "41",
                "passed": False,
                "elapsed_seconds": 2.0,
            },
        },
    }


def _install_instruction_precision_prompt_fixture(tmp_path: Path, monkeypatch) -> Path:
    prompt_dir = tmp_path / "reference-prompts"
    prompt_dir.mkdir()
    (prompt_dir / "instruction_precision.yaml").write_text(
        "prompts:\n"
        "  a:\n"
        "    prompt: 'A'\n"
        "    reference_answer: 'A ref'\n"
        "  b:\n"
        "    prompt: 'B'\n"
        "    reference_answer: 'B ref'\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        mod.REFERENCE_BACKED_COLLECTION_SOURCES,
        "instruction_precision",
        prompt_dir,
    )
    return prompt_dir


def test_pairwise_holdout_plan_selects_non_overlapping_cross_action_group(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    candidates = tmp_path / "candidates.jsonl"
    summary = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"

    assert (
        mod.main(
            [
                "--input",
                str(source),
                "--existing-manifest",
                str(manifest),
                "--existing-pairwise-jsonl",
                str(pairwise),
                "--candidates-jsonl",
                str(candidates),
                "--summary-json",
                str(summary),
                "--summary-md",
                str(summary_md),
                "--target-source-families",
                "seeding_eval",
                "--target-suites",
                "livecodebench",
                "--min-cross-action-candidate-groups",
                "1",
            ]
        )
        == 0
    )

    rows = [json.loads(line) for line in candidates.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(rows) == 1
    assert rows[0]["canonical_action"] == "coder_escalation"
    assert rows[0]["source_family"] == "seeding_eval"
    assert rows[0]["suite"] == "livecodebench"
    assert "prompt" not in rows[0]
    assert "response" not in rows[0]
    assert payload["decision"]["status"] == "expansion_plan_ready"
    assert payload["candidate_groups"] == 1
    assert payload["selected_groups"][0]["existing_manifest_actions"] == ["frontdoor"]
    assert payload["selected_groups"][0]["candidate_actions"] == ["coder_escalation"]
    assert "# Offline Reward Pairwise Holdout Expansion Plan" in summary_md.read_text(
        encoding="utf-8"
    )


def test_pairwise_holdout_plan_excludes_existing_pairwise_groups(tmp_path: Path) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    group_key = mod._record_group_key(
        source_path=source,
        offset=0,
        question_id="q-live",
        prompt_sha256=prompt_hash,
        expected_sha256=expected_hash,
    )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [{"group_key": group_key}])

    args = mod.build_parser().parse_args(
        [
            "--input",
            str(source),
            "--existing-manifest",
            str(manifest),
            "--existing-pairwise-jsonl",
            str(pairwise),
            "--candidates-jsonl",
            str(tmp_path / "candidates.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--target-source-families",
            "seeding_eval",
            "--target-suites",
            "livecodebench",
            "--min-cross-action-candidate-groups",
            "1",
        ]
    )

    candidates, summary = mod.build_plan(args)

    assert candidates == []
    assert summary["decision"]["status"] == "insufficient_non_overlapping_cross_action_candidates"
    assert summary["skipped_pairwise_overlap_groups"] == 1


def test_pairwise_holdout_cli_writes_negative_plan_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    group_key = mod._record_group_key(
        source_path=source,
        offset=0,
        question_id="q-live",
        prompt_sha256=prompt_hash,
        expected_sha256=expected_hash,
    )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [{"group_key": group_key}])
    candidates = tmp_path / "candidates.jsonl"
    summary = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"

    assert (
        mod.main(
            [
                "--input",
                str(source),
                "--existing-manifest",
                str(manifest),
                "--existing-pairwise-jsonl",
                str(pairwise),
                "--candidates-jsonl",
                str(candidates),
                "--summary-json",
                str(summary),
                "--summary-md",
                str(summary_md),
                "--target-source-families",
                "seeding_eval",
                "--target-suites",
                "livecodebench",
                "--min-cross-action-candidate-groups",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert candidates.read_text(encoding="utf-8") == ""
    assert payload["decision"]["status"] == "insufficient_non_overlapping_cross_action_candidates"
    assert payload["candidate_rows"] == 0
    assert "Candidate rows: `0`" in summary_md.read_text(encoding="utf-8")


def test_pairwise_holdout_any_mode_empty_suite_does_not_match_everything(
    tmp_path: Path,
) -> None:
    source = tmp_path / "3way_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [])
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    args = mod.build_parser().parse_args(
        [
            "--input",
            str(source),
            "--existing-manifest",
            str(manifest),
            "--existing-pairwise-jsonl",
            str(pairwise),
            "--candidates-jsonl",
            str(tmp_path / "candidates.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--target-source-families",
            "seeding_eval",
            "--target-suites",
            "",
            "--target-match-mode",
            "any",
            "--min-cross-action-candidate-groups",
            "1",
        ]
    )

    candidates, summary = mod.build_plan(args)

    assert candidates == []
    assert summary["stats"]["skipped_non_target_suite_record"] == 1


def test_pairwise_holdout_plan_filters_to_audit_collection_targets(tmp_path: Path) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "collection_targets": [
                    {
                        "stratum_field": "source_family",
                        "stratum_value": "seeding_eval",
                        "action_pair": "coder_escalation>frontdoor",
                        "current_rows": 2,
                        "current_direction_balance": 0.0,
                        "needs_direction": ["prefer other-side of coder_escalation>frontdoor"],
                        "prefer_hi": 0,
                        "prefer_lo": 2,
                        "suggested_min_rows": 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    args = mod.build_parser().parse_args(
        [
            "--input",
            str(source),
            "--existing-manifest",
            str(manifest),
            "--existing-pairwise-jsonl",
            str(pairwise),
            "--candidates-jsonl",
            str(tmp_path / "candidates.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--collection-targets-json",
            str(audit),
            "--target-source-families",
            "",
            "--target-suites",
            "",
            "--min-cross-action-candidate-groups",
            "1",
        ]
    )

    candidates, summary = mod.build_plan(args)

    assert len(candidates) == 1
    assert summary["decision"]["status"] == "insufficient_non_overlapping_cross_action_candidates"
    assert summary["collection_target_count"] == 1
    assert summary["matched_collection_target_counts"] == {}
    assert summary["candidate_presence_collection_target_counts"] == {
        "source_family:seeding_eval:coder_escalation>frontdoor": 1
    }
    assert summary["collection_target_match_metric"] == "source_binary_reward_directional_contrast"
    assert summary["unmatched_collection_targets"] == [
        "source_family:seeding_eval:coder_escalation>frontdoor"
    ]
    assert summary["candidate_contrast_groups"] == 0
    assert summary["source_record_requirements"] == [
        {
            "target": "source_family:seeding_eval:coder_escalation>frontdoor",
            "status": "needs_new_source_records",
            "stratum_field": "source_family",
            "stratum_value": "seeding_eval",
            "action_pair": "coder_escalation>frontdoor",
            "actions_to_evaluate_on_same_source_record": [
                "coder_escalation",
                "frontdoor",
            ],
            "target_preferred_actions": ["coder_escalation"],
            "needs_direction": ["prefer other-side of coder_escalation>frontdoor"],
            "current_rows": 2,
            "current_direction_balance": 0.0,
            "matched_candidate_groups": 0,
            "presence_candidate_groups": 1,
            "matched_candidate_group_metric": "source_binary_reward_directional_contrast",
            "suggested_min_rows": 20,
            "suggested_min_new_source_records": 20,
            "collection_priority": 0,
            "collection_priority_reason": "independent_holdout_source_family_blocker",
            "source_record_shape": (
                "one prompt/reference evaluated by every action in action_pair "
                "with role_results, rewards, suite, prompt, and expected fields"
            ),
            "runtime_gate_change_allowed": False,
        }
    ]
    assert (
        summary["collection_guidance"]["seeding_eval_command_template"]
        == "uv run python scripts/benchmark/seed_specialist_routing.py "
        "--suites <suite> --roles <actions_to_evaluate_on_same_source_record> "
        "--modes direct --sample-size <n> --max-tokens 1024 --strict-modes --dry-run "
        "--output <benchmarks/results/eval/seeding_a9_*.json>"
    )
    assert summary["collection_batches"] == [
        {
            "target": "source_family:seeding_eval:coder_escalation>frontdoor",
            "targets": ["source_family:seeding_eval:coder_escalation>frontdoor"],
            "expected_source_family": "seeding_eval",
            "suite_argument": "all",
            "roles_argument": ["coder_escalation", "frontdoor"],
            "modes_argument": ["direct"],
            "sample_size": 2,
            "max_tokens": 1024,
            "strict_modes": True,
            "question_source": "auto",
            "requested_new_source_records": 20,
            "estimated_new_source_records": 36,
            "sample_size_semantics": (
                "seed_specialist_routing.py interprets --sample-size as "
                "questions per suite; source-family targets with --suites "
                "all are downscaled to approximate the requested total."
            ),
            "collection_priority": 0,
            "collection_priority_reason": "independent_holdout_source_family_blocker",
            "durable_source_path": (
                "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/"
                "eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_"
                "<YYYYMMDDTHHMMSSZ>.json"
            ),
            "checkpoint_note": (
                "seed_specialist_routing.py also writes a seeding_*.jsonl "
                "checkpoint under benchmarks/results/eval; use the JSON path "
                "above when the target explicitly requires orchestrator_live_seed"
            ),
            "dry_run_semantics": (
                "--dry-run still performs scoring/evaluation; it only prevents "
                "reward injection into runtime memory."
            ),
            "command": (
                "uv run python scripts/benchmark/seed_specialist_routing.py "
                "--suites all --roles coder_escalation frontdoor "
                "--modes direct --sample-size 2 --max-tokens 1024 --strict-modes "
                "--dry-run --output "
                "/mnt/raid0/llm/epyc-inference-research/benchmarks/results/"
                "eval/seeding_a9_source_family_seeding_eval_coder_escalation_frontdoor_"
                "<YYYYMMDDTHHMMSSZ>.json"
            ),
            "can_run_during_active_autopilot": False,
            "reason": (
                "Consumes live model slots and should be run in a coordinated "
                "measurement window so A9 evidence is not mixed with W6/T2 accrual."
            ),
        }
    ]
    assert summary["selected_groups"][0]["matched_collection_targets"] == [
        {
            "stratum_field": "source_family",
            "stratum_value": "seeding_eval",
            "action_pair": "coder_escalation>frontdoor",
        }
    ]
    assert summary["selected_groups"][0]["binary_reward_contrast_collection_targets"] == []


def test_pairwise_holdout_counts_directional_binary_contrast(tmp_path: Path) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(
        source,
        [
            {
                "question_id": "q-live",
                "suite": "livecodebench",
                "prompt": "Solve task",
                "expected": "42",
                "role_results": {
                    "frontdoor:direct": {
                        "answer": "41",
                        "passed": False,
                        "elapsed_seconds": 1.0,
                    },
                    "coder_primary": {
                        "answer": "42",
                        "passed": True,
                        "elapsed_seconds": 2.0,
                    },
                },
            }
        ],
    )
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [])
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "collection_targets": [
                    {
                        "stratum_field": "source_family",
                        "stratum_value": "seeding_eval",
                        "action_pair": "coder_escalation>frontdoor",
                        "current_rows": 2,
                        "current_direction_balance": 0.0,
                        "needs_direction": ["prefer other-side of coder_escalation>frontdoor"],
                        "prefer_hi": 0,
                        "prefer_lo": 2,
                        "suggested_min_rows": 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    args = mod.build_parser().parse_args(
        [
            "--input",
            str(source),
            "--existing-manifest",
            str(manifest),
            "--existing-pairwise-jsonl",
            str(pairwise),
            "--candidates-jsonl",
            str(tmp_path / "candidates.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--collection-targets-json",
            str(audit),
            "--target-source-families",
            "",
            "--target-suites",
            "",
            "--min-cross-action-candidate-groups",
            "1",
        ]
    )

    candidates, summary = mod.build_plan(args)

    assert len(candidates) == 2
    target_key = "source_family:seeding_eval:coder_escalation>frontdoor"
    assert summary["decision"]["status"] == "expansion_plan_ready"
    assert summary["candidate_contrast_groups"] == 1
    assert summary["matched_collection_target_counts"] == {target_key: 1}
    contrast_targets = summary["selected_groups"][0]["binary_reward_contrast_collection_targets"]
    assert contrast_targets == [
        {
            "target": target_key,
            "stratum_field": "source_family",
            "stratum_value": "seeding_eval",
            "action_pair": "coder_escalation>frontdoor",
            "preferred_action": "coder_escalation",
            "rejected_action": "frontdoor",
            "contrast_metric": "source_binary_reward",
        }
    ]


def test_pairwise_holdout_plan_rejects_non_matching_audit_collection_targets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "collection_targets": [
                    {
                        "stratum_field": "source_family",
                        "stratum_value": "seeding_eval",
                        "action_pair": "architect_general>frontdoor",
                        "current_rows": 3,
                        "current_direction_balance": 0.0,
                        "needs_direction": ["prefer architect_general"],
                        "prefer_hi": 0,
                        "prefer_lo": 3,
                        "suggested_min_rows": 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    args = mod.build_parser().parse_args(
        [
            "--input",
            str(source),
            "--existing-manifest",
            str(manifest),
            "--existing-pairwise-jsonl",
            str(pairwise),
            "--candidates-jsonl",
            str(tmp_path / "candidates.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--collection-targets-json",
            str(audit),
            "--target-source-families",
            "",
            "--target-suites",
            "",
            "--min-cross-action-candidate-groups",
            "1",
        ]
    )

    candidates, summary = mod.build_plan(args)

    assert candidates == []
    assert summary["decision"]["status"] == "insufficient_non_overlapping_cross_action_candidates"
    assert summary["skipped_no_collection_target_pair_groups"] == 1
    assert summary["unmatched_collection_targets"] == [
        "source_family:seeding_eval:architect_general>frontdoor"
    ]
    assert summary["source_record_requirements"][0]["status"] == "needs_new_source_records"
    assert summary["source_record_requirements"][0]["target_preferred_actions"] == [
        "architect_general"
    ]
    assert summary["source_record_requirements"][0]["suggested_min_new_source_records"] == 20
    assert summary["source_record_requirements"][0]["collection_priority"] == 0
    assert (
        summary["source_record_requirements"][0]["collection_priority_reason"]
        == "independent_holdout_source_family_blocker"
    )


def test_pairwise_holdout_plan_uses_reference_backed_instruction_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prompt_dir = _install_instruction_precision_prompt_fixture(tmp_path, monkeypatch)
    source = tmp_path / "seeding_a9_suite_instruction_precision.json"
    record = {
        "question_id": "ifeval_1",
        "suite": "instruction_precision",
        "prompt": "Follow these instructions exactly",
        "expected": "",
        "role_results": {
            "architect_general:direct": {
                "answer": "Compliant response",
                "passed": True,
                "elapsed_seconds": 1.0,
            },
            "frontdoor:direct": {
                "answer": "Different compliant response",
                "passed": True,
                "elapsed_seconds": 1.0,
            },
        },
    }
    source.write_text(json.dumps({"results": [record]}), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [])
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "collection_targets": [
                    {
                        "stratum_field": "suite",
                        "stratum_value": "instruction_precision",
                        "action_pair": "architect_general>frontdoor",
                        "needs_direction": ["prefer architect_general"],
                        "current_rows": 0,
                        "suggested_min_rows": 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    args = mod.build_parser().parse_args(
        [
            "--input",
            str(source),
            "--existing-manifest",
            str(manifest),
            "--existing-pairwise-jsonl",
            str(pairwise),
            "--candidates-jsonl",
            str(tmp_path / "candidates.jsonl"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--collection-targets-json",
            str(audit),
            "--target-source-families",
            "",
            "--target-suites",
            "",
            "--min-cross-action-candidate-groups",
            "1",
        ]
    )

    candidates, summary = mod.build_plan(args)

    assert candidates == []
    assert summary["unavailable_collection_targets"] == {}
    assert summary["collection_guidance"]["reference_backed_suite_sources"] == {
        "instruction_precision": str(prompt_dir)
    }
    assert summary["source_record_requirements"][0]["status"] == "needs_new_source_records"
    assert summary["source_record_requirements"][0]["suggested_min_new_source_records"] == 20
    assert len(summary["collection_batches"]) == 1
    batch = summary["collection_batches"][0]
    assert batch["target"] == "suite:instruction_precision:architect_general>frontdoor"
    assert batch["targets"] == ["suite:instruction_precision:architect_general>frontdoor"]
    assert batch["suite_argument"] == "instruction_precision"
    assert batch["question_source"] == "yaml"
    assert batch["debug_prompts_dir"] == str(prompt_dir)
    assert batch["sample_size"] == 2
    assert batch["reference_source_prompt_count"] == 2
    assert batch["estimated_new_source_records"] == 2
    assert "--question-source yaml" in batch["command"]
    assert (
        f"--debug-prompts-dir {prompt_dir}" in batch["command"]
    )


def test_pairwise_holdout_collection_batches_prioritize_source_family_blockers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prompt_dir = _install_instruction_precision_prompt_fixture(tmp_path, monkeypatch)
    suite_requirement = {
        "target": "suite:instruction_precision:architect_general>frontdoor",
        "stratum_field": "suite",
        "stratum_value": "instruction_precision",
        "actions_to_evaluate_on_same_source_record": ["architect_general", "frontdoor"],
        "suggested_min_new_source_records": 20,
        "collection_priority": 2,
        "collection_priority_reason": "direction_balance_cleanup",
    }
    source_requirement = {
        "target": "source_family:seeding_eval:architect_general>frontdoor",
        "stratum_field": "source_family",
        "stratum_value": "seeding_eval",
        "actions_to_evaluate_on_same_source_record": ["architect_general", "frontdoor"],
        "suggested_min_new_source_records": 20,
        "collection_priority": 0,
        "collection_priority_reason": "independent_holdout_source_family_blocker",
    }

    batches = mod._collection_batches([suite_requirement, source_requirement])

    assert [batch["target"] for batch in batches] == [
        "source_family:seeding_eval:architect_general>frontdoor",
        "suite:instruction_precision:architect_general>frontdoor",
    ]
    assert batches[0]["sample_size"] == 2
    assert batches[0]["question_source"] == "auto"
    assert batches[0]["requested_new_source_records"] == 20
    assert batches[0]["estimated_new_source_records"] == 36
    assert batches[1]["sample_size"] == 2
    assert batches[1]["question_source"] == "yaml"
    assert batches[1]["debug_prompts_dir"] == str(prompt_dir)
    assert batches[1]["requested_new_source_records"] == 20
    assert batches[1]["reference_source_prompt_count"] == 2
    assert batches[1]["estimated_new_source_records"] == 2
    assert batches[0]["collection_priority_reason"] == (
        "independent_holdout_source_family_blocker"
    )


def test_pairwise_holdout_merges_reference_backed_suite_pairs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "instruction_precision.yaml").write_text(
        "prompts:\n"
        "  a:\n"
        "    prompt: 'A'\n"
        "    reference_answer: 'A ref'\n"
        "  b:\n"
        "    prompt: 'B'\n"
        "    reference_answer: 'B ref'\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        mod.REFERENCE_BACKED_COLLECTION_SOURCES,
        "instruction_precision",
        prompt_dir,
    )
    requirements = [
        {
            "target": "suite:instruction_precision:architect_general>coder_escalation",
            "stratum_field": "suite",
            "stratum_value": "instruction_precision",
            "action_pair": "architect_general>coder_escalation",
            "actions_to_evaluate_on_same_source_record": [
                "architect_general",
                "coder_escalation",
            ],
            "target_preferred_actions": ["architect_general"],
            "suggested_min_new_source_records": 20,
            "collection_priority": 2,
            "collection_priority_reason": "direction_balance_cleanup",
        },
        {
            "target": "suite:instruction_precision:architect_general>frontdoor",
            "stratum_field": "suite",
            "stratum_value": "instruction_precision",
            "action_pair": "architect_general>frontdoor",
            "actions_to_evaluate_on_same_source_record": [
                "architect_general",
                "frontdoor",
            ],
            "target_preferred_actions": ["frontdoor"],
            "suggested_min_new_source_records": 20,
            "collection_priority": 2,
            "collection_priority_reason": "direction_balance_cleanup",
        },
    ]

    batches = mod._collection_batches(requirements)

    assert len(batches) == 1
    batch = batches[0]
    assert batch["targets"] == [
        "suite:instruction_precision:architect_general>coder_escalation",
        "suite:instruction_precision:architect_general>frontdoor",
    ]
    assert batch["roles_argument"] == [
        "architect_general",
        "coder_escalation",
        "frontdoor",
    ]
    assert "--roles architect_general coder_escalation frontdoor" in batch["command"]
    assert batch["sample_size"] == 2
    assert batch["estimated_new_source_records"] == 2
    assert batch["reference_source_prompt_count"] == 2
    assert batch["question_source"] == "yaml"


def test_pairwise_holdout_negative_markdown_lists_source_record_requirements(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "collection_targets": [
                    {
                        "stratum_field": "source_family",
                        "stratum_value": "seeding_eval",
                        "action_pair": "architect_general>frontdoor",
                        "needs_direction": ["balance both directions"],
                        "current_rows": 2,
                        "suggested_min_rows": 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary_md = tmp_path / "summary.md"

    assert (
        mod.main(
            [
                "--input",
                str(source),
                "--existing-manifest",
                str(manifest),
                "--existing-pairwise-jsonl",
                str(pairwise),
                "--candidates-jsonl",
                str(tmp_path / "candidates.jsonl"),
                "--summary-json",
                str(tmp_path / "summary.json"),
                "--summary-md",
                str(summary_md),
                "--collection-targets-json",
                str(audit),
                "--target-source-families",
                "",
                "--target-suites",
                "",
                "--min-cross-action-candidate-groups",
                "1",
            ]
        )
        == 0
    )

    text = summary_md.read_text(encoding="utf-8")
    assert "## Source Record Requirements" in text
    assert "`source_family:seeding_eval:architect_general>frontdoor`" in text
    assert "preferred winners `['architect_general', 'frontdoor']`" in text


def test_pairwise_holdout_writes_guarded_collection_manifest_and_script(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    expected_hash = mod._hash_text("42")
    prompt_hash = mod._hash_text("Solve task")
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            _manifest_row(
                source_path=str(source),
                role_key="frontdoor:direct",
                expected_sha256=expected_hash,
                prompt_sha256=prompt_hash,
            )
        ],
    )
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "collection_targets": [
                    {
                        "stratum_field": "source_family",
                        "stratum_value": "orchestrator_live_seed",
                        "action_pair": "architect_general>frontdoor",
                        "needs_direction": ["prefer architect_general"],
                        "current_rows": 0,
                        "suggested_min_rows": 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    collection_manifest = tmp_path / "collection_manifest.json"
    collection_script = tmp_path / "collect.sh"

    assert (
        mod.main(
            [
                "--input",
                str(source),
                "--existing-manifest",
                str(manifest),
                "--existing-pairwise-jsonl",
                str(pairwise),
                "--candidates-jsonl",
                str(tmp_path / "candidates.jsonl"),
                "--summary-json",
                str(tmp_path / "summary.json"),
                "--collection-targets-json",
                str(audit),
                "--collection-manifest-json",
                str(collection_manifest),
                "--collection-script",
                str(collection_script),
                "--collection-timestamp",
                "20260628T120000Z",
                "--target-source-families",
                "",
                "--target-suites",
                "",
                "--min-cross-action-candidate-groups",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(collection_manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == mod.COLLECTION_MANIFEST_SCHEMA_VERSION
    assert payload["requires_active_autopilot_absent"] is True
    assert payload["autopilot_guard"]["refusal_exit_code"] == 75
    pipeline_text = "\n".join(payload["post_collection_pipeline"])
    assert "--artifact-scope candidate_only" in pipeline_text
    assert "offline_reward_pairwise_preference_contract_candidate_only_expanded_gap.jsonl" in pipeline_text
    assert "offline_reward_pairwise_preference_contract_expanded_gap.jsonl" not in pipeline_text
    assert payload["batch_count"] == 1
    batch = payload["batches"][0]
    assert batch["target"] == "source_family:orchestrator_live_seed:architect_general>frontdoor"
    assert batch["command_workdir"] == str(mod.DEFAULT_COLLECTION_WORKDIR)
    assert batch["collection_timestamp"] == "20260628T120000Z"
    assert batch["sample_size"] == 2
    assert batch["max_tokens"] == 1024
    assert batch["strict_modes"] is True
    assert batch["question_source"] == "auto"
    assert batch["requested_new_source_records"] == 20
    assert batch["estimated_new_source_records"] == 36
    assert "<YYYYMMDDTHHMMSSZ>" in batch["command_template"]
    assert "<YYYYMMDDTHHMMSSZ>" not in batch["command"]
    assert batch["durable_source_path"].endswith("20260628T120000Z.json")

    script_text = collection_script.read_text(encoding="utf-8")
    assert "pgrep -af 'scripts/autopilot/autopilot.py start'" in script_text
    assert "exit 75" in script_text
    # The script must cd into the repo root of the checkout that generated it
    # (worktree-invariant), not a hardcoded clone path.
    assert mod.DEFAULT_COLLECTION_WORKDIR == mod.PROJECT_ROOT
    assert f"cd {mod.PROJECT_ROOT}\n" in script_text
    assert "seeding_live_a9_source_family_orchestrator_live_seed" in script_text
    assert "20260628T120000Z" in script_text
    assert collection_script.stat().st_mode & 0o111


def test_pairwise_holdout_rejects_bad_collection_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "seeding_20260621_eval.jsonl"
    _write_jsonl(source, [_result_record()])
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [])
    pairwise = tmp_path / "pairs.jsonl"
    _write_jsonl(pairwise, [])

    assert (
        mod.main(
            [
                "--input",
                str(source),
                "--existing-manifest",
                str(manifest),
                "--existing-pairwise-jsonl",
                str(pairwise),
                "--candidates-jsonl",
                str(tmp_path / "candidates.jsonl"),
                "--summary-json",
                str(tmp_path / "summary.json"),
                "--collection-manifest-json",
                str(tmp_path / "collection_manifest.json"),
                "--collection-timestamp",
                "not-a-timestamp",
            ]
        )
        == 2
    )
