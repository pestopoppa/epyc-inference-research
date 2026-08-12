from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "autokernel_iq2_model_beliefs.py"
SPEC = importlib.util.spec_from_file_location("iq2_model_beliefs", PATH)
assert SPEC and SPEC.loader
producer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(producer)


def sha(tag: str) -> str:
    import hashlib
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


CAMPAIGN = "ak-iq2-model-confirmation"
CANDIDATE = "akc-iq2-one-row"
CLAIM = "akclaim-iq2model"
MODEL = "/models/Qwen3-IQ2_XXS.gguf"
SOURCE_COMMIT = sha("candidate-commit")[:40]
ANCHOR_COMMIT = sha("anchor-commit")[:40]
CANDIDATE_BINARY = sha("candidate-binary")
ANCHOR_BINARY = sha("anchor-binary")
CANDIDATE_LINKAGE = sha("candidate-linkage")
ANCHOR_LINKAGE = sha("anchor-linkage")
CANDIDATE_SOURCE = sha("candidate-source")
HOST_RECEIPT = "host-health-iq2-model"
NOW = "2026-08-11T20:00:00+00:00"


def anchor() -> dict:
    return {
        "source_commit": ANCHOR_COMMIT,
        "binary_sha256": ANCHOR_BINARY,
        "linkage_sha256": ANCHOR_LINKAGE,
        "measurement_event_ids": ["ake-anchor-iq2"],
    }


def claim_receipt() -> dict:
    root = "/mnt/raid0/llm/tmp"
    regions = ["q0", "q1", "q2", "q3"]
    paths = sorted(
        [f"{root}/cpu_region.GLOBAL.{region}.lock" for region in regions]
        + [f"{root}/cpu_region.autokernel.{region}.lock" for region in regions]
    )
    return {
        "schema": "epyc.autokernel.cpu_region_claim_receipt.v1",
        "claim_id": CLAIM,
        "role": "autokernel",
        "roles": ["autokernel"],
        "cpu_list": "0-191",
        "physical_core_list": "0-95",
        "regions": regions,
        "lock_paths": paths,
        "lock_root": root,
        "state": "held",
        "holder_pid": 1234,
        "holder_start_ticks": 5678,
        "holder_boot_id": "00000000-0000-0000-0000-000000000001",
        "host": "Beelzebub",
        "holder_label": "iq2-model-confirmation",
        "purpose": "IQ2_XXS model confirmation",
        "campaign_id": CAMPAIGN,
        "acquired_at": "2026-08-11T19:55:00+00:00",
        "expires_at": "2026-08-11T21:55:00+00:00",
        "released_at": "2026-08-11T20:05:00+00:00",
        "reclaimed_from": None,
    }


def candidate_record(event_ids: list[str]) -> dict:
    return {
        "schema": "epyc.autokernel.candidate.v1",
        "candidate_id": CANDIDATE,
        "campaign_id": CAMPAIGN,
        "proposal_id": "akp-iq2-one-row",
        "parent_candidate_id": None,
        "worktree": {
            "path": "/work/candidate",
            "branch": "experimental-v9-iq2-one-row",
            "source_commit": SOURCE_COMMIT,
            "clean": True,
        },
        "source_snapshot": {
            "snapshot_sha256": CANDIDATE_SOURCE,
            "patch_bundle_sha256": sha("candidate-patch"),
        },
        "ancestry": {
            "production_base_commit": ANCHOR_COMMIT,
            "is_descendant_of_production_base": True,
            "proof": "git merge-base --is-ancestor",
        },
        "build": {
            "toolchain": "cmake+clang",
            "compiler": "clang 20",
            "command": "cmake --build build --target llama-bench",
            "build_dir": "/work/candidate/build",
            "log_path": "/evidence/build.log",
            "log_sha256": sha("build-log"),
        },
        "artifacts": {
            "binary_sha256": CANDIDATE_BINARY,
            "linkage_sha256": CANDIDATE_LINKAGE,
            "library_sha256s": {"libggml-cpu.so": sha("candidate-lib")},
        },
        "dispatch": {"feature_flags": ["GGML_IQK=1"], "dispatch_predicate": "IQ2_XXS n==1"},
        "affected_surface": {
            "derived_sha256": sha("surface"),
            "traced_sha256": sha("surface"),
            "reconciled": True,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 2},
        "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": sha("evaluator")},
        "receipts": {"host_receipt": HOST_RECEIPT, "resource_claim_receipt": CLAIM},
        "storage": {"footprint_gb": 0.1, "durability_class": "durable_untracked"},
        "evaluation_event_ids": event_ids,
        "derived_verdicts": {event_id: "pass" for event_id in event_ids},
        "controller": {
            "provider": "operator",
            "model_id": "codex",
            "effort": "high",
            "prompt_bundle_sha256": sha("prompt"),
        },
        "champion_status": "none",
        "status": "evaluating",
        "supersession_reason": None,
        "created_at": NOW,
    }


def execution_receipt(*, lane: str, arm: str) -> dict:
    recipe = producer.LANES[lane]["recipe_id"]
    if arm == "candidate":
        source_root = "/work/candidate"
        binary_path = "/work/candidate/build/bin/llama-bench"
        binary_sha256 = CANDIDATE_BINARY
        linkage = "/work/candidate/build/bin"
    else:
        source_root = "/work/anchor"
        binary_path = "/work/anchor/build/bin/llama-bench"
        binary_sha256 = ANCHOR_BINARY
        linkage = "/work/anchor/build/bin"
    return {
        "runner_id": "autokernel.execution.microbench/v1",
        "recipe_id": recipe,
        "registry_id": "ak-recipe-registry/v1",
        "arm": arm,
        "constructor_id": recipe,
        "constructor_sha256": sha(f"constructor-{lane}"),
        "argv_sha256": sha(f"argv-{lane}-{arm}"),
        "argv": [binary_path, "-m", MODEL, "-o", "json"],
        "recipe_env": {"GGML_IQK": "1", "LD_LIBRARY_PATH": linkage},
        "params": {"model": MODEL, "reps": 2},
        "env": {"GGML_IQK": "1", "LD_LIBRARY_PATH": linkage},
        "env_sha256": sha(f"env-{lane}-{arm}"),
        "binary_path": binary_path,
        "binary_sha256": binary_sha256,
        "binary_size": 123456,
        "source_root": source_root,
        "library_path": linkage,
        "resolved_at": NOW,
    }


def block(*, lane: str, index: int, anchor_samples: list[float],
          candidate_samples: list[float]) -> dict:
    order = "anchor_first" if index % 2 == 0 else "candidate_first"
    unit_id = f"qwen3-iq2-xxs:{lane}"
    paired = [
        index, unit_id, "confirmation", order, "base", None, NOW,
        anchor_samples, candidate_samples,
    ]

    def invocation(arm: str, samples: list[float]) -> dict:
        receipt = execution_receipt(lane=lane, arm=arm)
        return {
            "arm": arm,
            "block_index": index,
            "position": 0 if (arm == "anchor") == (order == "anchor_first") else 1,
            "recipe": lane,
            "receipt": receipt,
            "row": {"model_filename": MODEL, "samples_ts": samples,
                    "avg_ts": sum(samples) / len(samples)},
            "samples": samples,
            "claim": {"claim_id": CLAIM, "outcome": "PASS"},
            "checks": [["output_matches_recipe", {"outcome": "PASS", "reasons": []}]],
            "spawn": {"returncode": 0, "timed_out": False},
        }

    return {
        "plan": {"block_index": index, "unit_id": unit_id, "order": order,
                 "stratum": "confirmation", "segment": "base"},
        "invocations": [
            invocation("anchor", anchor_samples),
            invocation("candidate", candidate_samples),
        ],
        "host_state_open": {},
        "host_state_close": {},
        "package_power": {},
        "paired_block": paired,
        "checks": [],
        "refusals": [],
        "complete": True,
    }


def raw_vector(lane: str) -> dict:
    values = {
        "tg": [([10.0, 12.0], [11.0, 13.0]), ([14.0, 16.0], [15.0, 17.0])],
        "pp": [([100.0, 102.0], [103.0, 105.0]), ([104.0, 106.0], [107.0, 109.0])],
    }[lane]
    blocks = [
        block(lane=lane, index=index, anchor_samples=a, candidate_samples=c)
        for index, (a, c) in enumerate(values)
    ]
    return {
        "schema": "epyc.autokernel.microbench_raw_vector.v1",
        "runner_id": "autokernel.execution.microbench/v1",
        "recipe_id": producer.LANES[lane]["recipe_id"],
        "candidate_id": CANDIDATE,
        "attempt": 0,
        "campaign_seed_sha256": sha("seed"),
        "order_schedule": {},
        "segment": "base",
        "extension_round": None,
        "anchor": "bound",
        "anchor_identity": anchor(),
        "scope_denominator": {
            "machine_subset": "full", "numa_nodes": [], "devices": [], "cores": 96,
        },
        "scope_render": "full host",
        "started_at": NOW,
        "ended_at": "2026-08-11T20:04:00+00:00",
        "complete": True,
        "order_control": {"outcome": "PASS", "reasons": []},
        "refusals": [],
        "checks": [],
        "candidate_receipt": execution_receipt(lane=lane, arm="candidate"),
        "anchor_receipt": execution_receipt(lane=lane, arm="anchor"),
        "unit_receipts": {},
        "claim_attestations": [
            {"claim_id": CLAIM, "outcome": "PASS", "cpu_list": "0-191"}
            for _ in range(4)
        ],
        "blocks": blocks,
    }


def event(*, event_id: str, tier: str, metric: str, estimate: float,
          raw_samples: list, transfer: list[dict]) -> dict:
    return {
        "schema": "epyc.autokernel.evaluation_event.v5",
        "event_id": event_id,
        "campaign_id": CAMPAIGN,
        "candidate_id": CANDIDATE,
        "tier": tier,
        "backend": "llama_cpu",
        "device_state": None,
        "change_class": "arithmetic",
        "anchor_tier": "T1a",
        "transfer_ratio_to": transfer,
        "claim_grammar": {
            "category": "CANDIDATE",
            "protocol_id": "P-AK-SEARCH-1/v1",
            "metric": metric,
            "metric_direction": "higher_better" if tier == "T2" else "lower_better",
            "reps": 2,
            "attestation_ref": f"akcap:{event_id}",
        },
        "evaluator": {"id": "P-AK-SEARCH-1/v1", "bundle_sha256": sha("evaluator")},
        "artifact": {
            "source_sha256": CANDIDATE_SOURCE,
            "binary_sha256": CANDIDATE_BINARY,
            "linkage_sha256": CANDIDATE_LINKAGE,
        },
        "anchor": anchor(),
        "scope_manifest_sha256": sha("scope"),
        "host_receipt": HOST_RECEIPT,
        "resource_claim_receipt": CLAIM,
        "co_residency": "single",
        "correctness": {},
        "quality": {},
        "stability": {},
        "mechanism": {},
        "scope_denominator": {
            "machine_subset": "full", "numa_nodes": [], "devices": [], "cores": 96,
        },
        "determinism": {"class": "bitwise_stable", "same_seed_repeat_runs": 2},
        "performance": {
            "raw_samples": raw_samples,
            "raw_samples_ref": f"evidence:{event_id}",
            "paired_blocks": 2,
            "estimate": estimate,
            "delta_display": str(estimate),
            "uncertainty": None,
            "search_discipline": {
                "search_grade": {"satisfied": True, "failed": []},
                "void_findings": [],
                "effect_resolution": "improvement",
                "speed_rank_admissible": True,
            },
        },
        "integrity_flags": [],
        "status": "pass",
        "supersedes": [],
        "created_at": NOW,
    }


def lane(lane_name: str) -> dict:
    t1_id = f"ake-iq2-{lane_name}-t1"
    t2_id = f"ake-iq2-{lane_name}-t2"
    raw = raw_vector(lane_name)
    t1 = event(
        event_id=t1_id, tier="T1a", metric="iq2_xxs_backend_op_time_us",
        estimate=-0.1, raw_samples=[1.0, 0.9], transfer=[],
    )
    t2 = event(
        event_id=t2_id, tier="T2", metric=producer.LANES[lane_name]["metric"],
        estimate=0.05, raw_samples=[row["paired_block"] for row in raw["blocks"]],
        transfer=[{
            "event_id": t1_id, "tier": "T1a",
            "source_effect": 0.05, "target_effect": -0.1, "ratio": -0.5,
        }],
    )
    return {"t1_event": t1, "t2_event": t2, "raw_vectors": [raw]}


def receipt() -> dict:
    lanes = {name: lane(name) for name in ("tg", "pp")}
    event_ids = [
        lanes[name][key]["event_id"]
        for name in ("tg", "pp") for key in ("t1_event", "t2_event")
    ]
    return {
        "schema": producer.SCHEMA,
        "status": "complete",
        "campaign_id": CAMPAIGN,
        "candidate_id": CANDIDATE,
        "created_at": NOW,
        "ended_at": "2026-08-11T20:05:00+00:00",
        "model_identity": {
            "model_id": "Qwen3-IQ2-XXS",
            "path": MODEL,
            "sha256": sha("model"),
            "quantization": "IQ2_XXS",
        },
        "candidate_record": candidate_record(event_ids),
        "anchor_identity": anchor(),
        "resource_claim_receipt": claim_receipt(),
        "lanes": lanes,
    }


def test_finalizer_emits_distinct_tg_pp_rows_for_both_matched_arms() -> None:
    source = receipt()
    finalized = producer.finalize_receipt(source, producer_sha256="f" * 64)
    assert "belief_measurements" not in source
    rows = finalized["belief_measurements"]
    assert len(rows) == 4
    assert {(row["extra"]["lane"], row["extra"]["arm"]) for row in rows} == {
        ("tg", "anchor"), ("tg", "candidate"),
        ("pp", "anchor"), ("pp", "candidate"),
    }
    assert all(row["metric_direction"] == "higher_better" for row in rows)
    assert all(row["reps"] == 4 for row in rows)
    assert all(row["extra"]["scored_blocks"] == 2 for row in rows)
    assert all(row["extra"]["model_identity"]["quantization"] == "IQ2_XXS"
               for row in rows)
    assert next(row for row in rows if row["measurement_id"] ==
                "iq2_xxs_model_tg_candidate_median_tokens_per_s")["value"] == 14.0
    assert next(row for row in rows if row["measurement_id"] ==
                "iq2_xxs_model_pp_anchor_median_tokens_per_s")["value"] == 103.0
    without_self = dict(finalized)
    without_self.pop("self_sha256")
    assert finalized["self_sha256"] == producer._content_sha256(without_self)
    for row in rows:
        expected = copy.deepcopy(row)
        digest = expected["extra"].pop("self_sha256")
        assert digest == producer._content_sha256(expected)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda value: value.update(status="failed"), "status"),
        (lambda value: value["resource_claim_receipt"].update(released_at=None), "released"),
        (lambda value: value["model_identity"].update(quantization="IQ2_XS"), "IQ2_XXS"),
        (lambda value: value["lanes"]["tg"]["t2_event"]["performance"]
         ["search_discipline"].update(speed_rank_admissible=False), "speed rank"),
        (lambda value: value["lanes"]["tg"]["raw_vectors"][0].update(complete=False),
         "incomplete"),
        (lambda value: value["lanes"]["tg"]["raw_vectors"][0]["candidate_receipt"]
         ["params"].update(model="/models/other.gguf"), "model_identity"),
        (lambda value: value["lanes"]["pp"]["t2_event"]["performance"]
         .update(raw_samples=[["tampered"]]), "raw_samples"),
        (lambda value: value["lanes"]["tg"]["t2_event"].update(transfer_ratio_to=[]),
         "transfer binding"),
    ],
)
def test_finalizer_refuses_incomplete_mixed_or_unadmitted_material(mutate, message) -> None:
    value = receipt()
    mutate(value)
    with pytest.raises(producer.ReceiptRefused, match=message):
        producer.finalize_receipt(value, producer_sha256="f" * 64)


def test_old_micro_receipt_cannot_be_retrofitted_or_re_finalized() -> None:
    with pytest.raises(producer.ReceiptRefused, match="schema"):
        producer.finalize_receipt(raw_vector("tg"), producer_sha256="f" * 64)
    finalized = producer.finalize_receipt(receipt(), producer_sha256="f" * 64)
    with pytest.raises(producer.ReceiptRefused, match="write-once"):
        producer.finalize_receipt(finalized, producer_sha256="f" * 64)


def test_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    finalized = producer.finalize_receipt(receipt(), producer_sha256="f" * 64)
    producer.write_json_atomic(output, finalized)
    assert output.is_file()
    with pytest.raises(FileExistsError):
        producer.write_json_atomic(output, finalized)
