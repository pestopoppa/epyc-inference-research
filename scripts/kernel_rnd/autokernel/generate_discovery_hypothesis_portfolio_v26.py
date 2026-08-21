#!/usr/bin/env python3
"""Generate the reviewed v26 single-frame discovery portfolio from v2 history."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.kernel_rnd.autokernel import hypothesis_portfolio


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "discovery_hypothesis_portfolio_v2.json"
TARGET = HERE / "discovery_hypothesis_portfolio_v26.json"
FRAME = "frame-current-qwen05b-q4km-tg128"
EVIDENCE = "ev-current-qwen05-profile"
V25_ROOT = Path(
    "/mnt/raid0/llm/autokernel/deployments/"
    "gpu-discovery-quant-ladder-occupancy-v25")
V25_SOURCE_MANIFESTS = (
    (2, "5d23961ff6754af5374ff70d3467f400ca0c24bdff222dec8267f004ff934826",
     "a0ecc2f7301dd4d708b1c6aef332a355b60044fe900e2ff0ecfa65ee36dc3d2a"),
    (4, "940db9dffb2205def5f81de7de39273f6f65266a643dd8e9e3bcc9d6fa687a14",
     "9bd95dcc8d14f5261776ecda073fea64ffeb9dff7257caf4f47a4041453c928f"),
    (6, "eb6b41546af7e3f5b76f7a045d311ac58a3e7439c6ec776eede4f32fd7c5bcb8",
     "180bc1fb4b98b089951a4f88d4351d94b6adec4ac657dec0e8775986127f0093"),
    (8, "0abeaba287efc30b280393c1c4356f3aa8a3d9dd77b612b1333ac227dce8bf0d",
     "6f92ff088704a8f204c1f999b90109b492be5b6c3761c9b86da814e5205c93ad"),
    (12, "a3b8d836effdfdf3eb976219604e97c1a596041a8a028273d52a9180a848508c",
     "ead43ad12a46b8cc8116ecd871c81354e4bb1be770d72232d2c6d35252b32cae"),
    (13, "b92b9bce27df5506636b151d9e199751671a452cf09e60d2275fa33449970dc7",
     "1a507529b68f1ad431eaea20b6c13c88c1e48f9a654e639480def706a6a0fa9d"),
    (14, "c346fd7f19bd977bb2bb8f9c7d4c5a8c644693570d94aa9929846e17e74f6bb4",
     "53f1df458e8161c4668835615eae8af19497f2735956f0e41f924bfee04e3b9f"),
)


def _signature(template: str, index: int, literal: str, calls: int, grid: int,
               workgroup: int, lds_bytes: int) -> dict[str, object]:
    return {
        "kernel_literal": literal,
        "calls": calls,
        "grid": grid,
        "workgroup": workgroup,
        "lds_bytes": lds_bytes,
        "route_id": f"{template}.anchor.{index}",
    }


def _record(*, rank: int, hypothesis_id: str, title: str, statement: str,
            quant: str, shape: str, files: list[str], symbols: list[str],
            template: str, mechanism: str, change_class: str,
            signatures: list[dict[str, object]], share: float,
            expected_gain: tuple[float, float], cost: str, risk: str,
            falsifiers: list[str], notes: list[str], evidence_refs: list[str],
            next_action: str) -> dict[str, object]:
    facets = {
        "mechanism": mechanism,
        "ops": [shape],
        "files": files,
        "symbols": symbols,
        "change_class": change_class,
    }
    policy = {
        "metric": "decode_tokens_per_s",
        "frame_id": FRAME,
        "continuation_floor_pct": 0.2,
        "nomination_floor_pct": 0.5,
        "required_replications": 2,
        "sign_policy": "all_positive",
        "conflict_policy": "retain_inconclusive",
        "max_distinct_candidates": 2,
        "terminal_rule": "retire",
        "effect_unit": "relative_percent",
        "min_replication_effect_pct": 0.1,
        "max_replication_spread_pct": 1,
    }
    return {
        "hypothesis_id": hypothesis_id,
        "title": title,
        "status": "queued",
        "statement": statement,
        "primary_falsifier": falsifiers[0],
        "regime": {
            "backend": "hip", "phase": "decode", "batch": 1,
            "architecture": "gfx90a", "model_or_frame": FRAME,
            "quant": quant, "shape": shape,
            "measurement_graphs": False, "target_runtime_graphs": True,
        },
        "target": {
            "frame_ids": [FRAME], "source_files": files,
            "source_symbols": symbols, "template_intent": template,
        },
        "dispatch_anchors": [{
            "frame_id": FRAME, "signatures": signatures,
            "total_calls": sum(int(row["calls"]) for row in signatures),
            "aggregation": "exact_signatures",
            "selection": "All and only exact current-frame routes owned by this reviewed question.",
            "evidence_ref": EVIDENCE, "excluded_signatures": [],
        }],
        "mechanism": {
            "facets": facets,
            "fingerprint_sha256": hypothesis_portfolio.mechanism_fingerprint(facets),
        },
        "falsifiers": falsifiers,
        "evidence_refs": evidence_refs,
        "interactions": [],
        "portability": {
            "level": "exact_frame", "source_frames": [FRAME],
            "target_frames": [FRAME],
            "constraints": [
                "Exact current-frame dispatch only",
                "Preserve reviewed launch geometry and numeric contract",
            ],
            "required_validation": [
                "Full operation correctness", "Exact route attribution",
                "Graphs-on paired whole-model screen",
            ],
        },
        "priority": {
            "rank": rank, "tier": "P0" if rank <= 3 else "P1",
            "device_time_share_pct_range": [share, share],
            "rationale": (
                "Ranked by exact current-frame device share, mechanism signal, "
                "implementation cost, and correctness blast radius."),
        },
        "expected_value": {
            "metric": "decode_tokens_per_s", "direction": "higher_better",
            "expected_relative_gain_pct_range": list(expected_gain),
            "device_time_ceiling_pct": share,
            "device_time_ceiling_frame_id": FRAME,
            "current_bundle_plausible_gain_ceiling_pct": share,
            "basis": (
                "Exact graphs-off route share is a ceiling and routing fact only; "
                "graphs-on paired throughput decides continuation."),
        },
        "implementation": {"cost": cost, "risk": risk, "notes": notes},
        "stop_rule": falsifiers,
        "current_bundle_eligibility": {
            "eligible": True, "template_ids": [template],
            "blocking_conditions": [],
            "reason": (
                "Exact current route, source scope, falsifier, and route-scoped "
                "reviewed template are sealed."),
        },
        "lifecycle": {
            "maturity": "characterized", "next_action": next_action,
            "candidate_identity": None, "diagnostic_identity": None,
        },
        "decision_policy": policy,
        "epistemic": {
            "grade": "profile_routing", "confidence": "high",
            "limitations": [
                "Routing and device share do not establish candidate speedup.",
                "Only a governed paired screen can spend scientific budget.",
            ],
        },
        "record_version": 1,
        "provenance": {
            "introduced_at": "2026-08-21T10:00:00Z",
            "introduced_by": "autokernel-v26-successor",
            "origin": "v25-cost-and-capacity-audit",
            "note": "Fresh route-scoped v26 question; v25 terminals remain noneligible history.",
            "supersedes": None,
        },
    }


def generate() -> dict[str, object]:
    body = json.loads(SOURCE.read_text(encoding="utf-8"))
    body["corpus_id"] = "gpu-decode-v26-20260821"
    body["generated_at"] = "2026-08-21T10:00:00Z"
    bundle = body["current_bundle"]
    bundle["bundle_id"] = "gpu-source-templates-v3-qwen05b-tg128"
    bundle["template_catalog_version"] = "gpu-source-templates-v3"
    bundle["template_ids"] = sorted(set(bundle["template_ids"]) | {
        "cuda-fattn-combine-v1", "cuda-fattn-gqa7-common-v1",
        "cuda-vecdotq-q4k-v1", "cuda-vecdotq-q6k-v1",
    })
    bundle["eligibility_semantics"] = (
        "Eligible means one exact current-frame route and one route-scoped reviewed "
        "template; scheduling minimizes prior exposure before using ROI rank.")
    body["evidence"].extend([
        {
            "evidence_id": "ev-v25-terminal-state",
            "path": ("/mnt/raid0/llm/autokernel/deployments/"
                     "gpu-discovery-quant-ladder-occupancy-v25/state/state.json"),
            "sha256": "7ce6e5561572390e0a1a31ff8a059be3b68c8cfc809a9233c2e22a8ca730ef3c",
            "authority": "governance_snapshot",
            "temporal_status": "current_v9",
            "claims": [
                "Complete v25 controller state with exact terminal, skip, and candidate identities",
                "Predecessor authority for cross-campaign replay refusal",
            ],
        },
        {
            "evidence_id": "ev-v25-terminal-journal",
            "path": ("/mnt/raid0/llm/autokernel/deployments/"
                     "gpu-discovery-quant-ladder-occupancy-v25/state/journal/events.jsonl"),
            "sha256": "a715dbbf8a8e089ea9e356339ceaf8f007bf6191ee0ea699d445c1560ddc5b69",
            "authority": "governance_snapshot",
            "temporal_status": "current_v9",
            "claims": [
                "Exact v25 terminal event chain through portfolio exhaustion and discovery complete",
                "Binds the final state semantic digest used by successor carry-forward",
            ],
        },
    ])
    body["evidence"].extend({
        "evidence_id": f"ev-v25-source-manifest-turn{turn:02d}",
        "path": str(V25_ROOT / "operations" / operation_key /
                    "source-manifest.json"),
        "sha256": source_manifest_sha256,
        "authority": "governance_snapshot",
        "temporal_status": "current_v9",
        "claims": [
            f"Exact v25 turn {turn} source candidate manifest",
            "Derives predecessor patch and cross-campaign semantic replay identity",
        ],
    } for turn, operation_key, source_manifest_sha256 in V25_SOURCE_MANIFESTS)

    old = body["hypotheses"]
    for row in old:
        row["priority"]["rank"] += 6
    outcomes = {
        "akh-v2-q5-type-specific-dequant":
            "v25 nominated this family after two scientific screens; discovery replay is forbidden.",
        "akh-v2-q8-quantizer-new-mechanism":
            "v25 retired this family after three distinct nonpositive scientific screens.",
        "akh-v2-fa-gqa7-pair-tail":
            "v25 bounded-skipped the old one-file authoring authority after three critic refusals.",
        "akh-v2-rms-direct-load-reduction":
            "v25 bounded-skipped this generic mechanism after three source-scope refusals; v26 uses a distinct broadcast mechanism.",
    }
    for row in old:
        reason = outcomes.get(row["hypothesis_id"])
        if reason is not None:
            for evidence_id in ("ev-v25-terminal-state", "ev-v25-terminal-journal"):
                if evidence_id not in row["evidence_refs"]:
                    row["evidence_refs"].append(evidence_id)
            row["current_bundle_eligibility"] = {
                "eligible": False,
                "template_ids": row["current_bundle_eligibility"]["template_ids"],
                "blocking_conditions": [reason],
                "reason": reason,
            }

    q4 = _signature("cuda-vecdotq-q4k-v1", 0,
                    "mul_mat_vec_q<(ggml_type)12,1,true,false>",
                    1548, 114688, 128, 512)
    q6 = _signature("cuda-vecdotq-q6k-v1", 0,
                    "mul_mat_vec_q<(ggml_type)14,1,true,false>",
                    1548, 114688, 128, 512)
    rms = _signature("cuda-norm-v2", 0,
                     "rms_norm_f32<256,true,false>", 6321, 256, 256, 512)
    rope = [
        _signature("cuda-rope-v2", 0,
                   "rope_neox<true,false,float,__half>", 3096, 512, 256, 0),
        _signature("cuda-rope-v2", 1,
                   "rope_neox<true,false,float,float>", 3096, 3584, 256, 0),
    ]
    combine = _signature("cuda-fattn-combine-v1", 0,
                         "flash_attn_combine_results<64>", 3096, 896, 64, 512)
    gqa = [
        _signature("cuda-fattn-gqa7-common-v1", 0,
                   "flash_attn_tile<64,64,2,1,false>", 3096, 7168, 64, 5120),
        _signature("cuda-fattn-gqa7-common-v1", 1,
                   "flash_attn_combine_results<64>", 3096, 896, 64, 512),
    ]
    fresh = [
        _record(
            rank=1, hypothesis_id="akh-v26-q4k-branchless-sixbit-scale",
            title="Q4_K lane-uniform six-bit scale/min selection",
            statement=("Packed selection can remove the lane-uniform j<2 scale/min "
                       "branch while preserving dp4a and exact launch geometry."),
            quant="Q4_K_M", shape="type12-mmvq",
            files=["ggml/src/ggml-cuda/vecdotq.cuh"],
            symbols=["vec_dot_q4_K_q8_1", "vec_dot_q4_K_q8_1_impl_vmmq"],
            template="cuda-vecdotq-q4k-v1",
            mechanism="q4k_lane_uniform_sixbit_scale_select", change_class="arithmetic",
            signatures=[q4], share=3.8994134849336506, expected_gain=(0.1, 0.5),
            cost="low", risk="medium",
            falsifiers=["Any Q4_K correctness mismatch falsifies the candidate",
                        "Exact type-12 duration is non-positive",
                        "Graphs-on throughput direction is non-positive"],
            notes=["Clean reconstruction only", "Dirty diagnostic is a design prior, not claim evidence"],
            evidence_refs=[EVIDENCE, "ev-q4k-branchless-dirty-r3",
                           "ev-fable5-mi210-lever-matrix", "ev-research-loop-snapshot"],
            next_action="Author one clean packed-select variant inside the two reviewed Q4_K symbols."),
        _record(
            rank=2, hypothesis_id="akh-v26-rms-scale-broadcast",
            title="RMSNorm single-scale shared broadcast",
            statement=("Lane zero can compute the identical post-reduction mean and rsqrt once "
                       "and broadcast the scale through existing shared storage."),
            quant="Q4_K_M", shape="rms_norm_f32",
            files=["ggml/src/ggml-cuda/norm.cu"], symbols=["rms_norm_f32"],
            template="cuda-norm-v2", mechanism="rms_lane0_scale_shared_broadcast",
            change_class="arithmetic", signatures=[rms], share=10.734949639102124,
            expected_gain=(0.2, 1.5), cost="low", risk="medium",
            falsifiers=["Any RMSNorm correctness mismatch falsifies the candidate",
                        "Exact RMSNorm duration is non-positive",
                        "Launch geometry or 512-byte LDS changes",
                        "Graphs-on throughput direction is non-positive"],
            notes=["Distinct from v25 x-cache patches", "Retain four-wave workgroup-256 geometry"],
            evidence_refs=[EVIDENCE, "ev-source-plan-v1", "ev-v25-terminal-state",
                           "ev-v25-terminal-journal"],
            next_action="Author the single-scale shared-broadcast mechanism after source-scope validation."),
        _record(
            rank=3, hypothesis_id="akh-v26-rope-neox-index-strength-reduction",
            title="RoPE NEOX exact index strength reduction",
            statement=("Exact quotient, remainder, and pair indices can be reused across both "
                       "current NEOX routes without changing powf or trigonometric semantics."),
            quant="Q4_K_M", shape="rope-neox-two-route",
            files=["ggml/src/ggml-cuda/rope.cu"],
            symbols=["rope_neox", "rope_neox_cuda"], template="cuda-rope-v2",
            mechanism="rope_neox_exact_index_strength_reduction", change_class="arithmetic",
            signatures=rope, share=8.61269905295275, expected_gain=(0.1, 1.0),
            cost="low", risk="medium",
            falsifiers=["Any RoPE correctness mismatch falsifies the candidate",
                        "Either exact NEOX route drifts",
                        "Combined exact duration is non-positive",
                        "Graphs-on throughput direction is non-positive"],
            notes=["No approximate reciprocal", "No fast-math substitution", "Preserve workgroup 256"],
            evidence_refs=[EVIDENCE, "ev-source-plan-v1", "ev-research-loop-snapshot"],
            next_action="Author exact integer index reuse across the two sealed NEOX routes."),
        _record(
            rank=4, hypothesis_id="akh-v26-fa-combine-wave-normalization",
            title="Flash-attention combine wave-cooperative normalization",
            statement=("For the exact 64-block combine route, one wave lane per metadata block "
                       "can reduce max and denominator and reuse scales via existing shared memory."),
            quant="Q4_K_M", shape="fattn-combine-d64",
            files=["ggml/src/ggml-cuda/fattn-common.cuh"],
            symbols=["flash_attn_combine_results"], template="cuda-fattn-combine-v1",
            mechanism="fattn_combine_wave_cooperative_metadata_normalization",
            change_class="arithmetic", signatures=[combine], share=3.719734811288374,
            expected_gain=(0.1, 0.8), cost="medium", risk="medium",
            falsifiers=["Any flash-attention correctness mismatch falsifies the candidate",
                        "Exact combine route geometry drifts",
                        "Exact combine duration is non-positive",
                        "Graphs-on throughput direction is non-positive"],
            notes=["512-byte LDS proves 64 metadata blocks", "Preserve output ordering and tile route"],
            evidence_refs=[EVIDENCE, "ev-source-plan-v1"],
            next_action="Author one exact 64-lane metadata normalization specialization."),
        _record(
            rank=5, hypothesis_id="akh-v26-q6k-packed-decode",
            title="Q6_K packed decode address-arithmetic reduction",
            statement=("Precomputed lane offsets and shifts can reduce packed ql/qh/sign/scale "
                       "address arithmetic while preserving dp4a and two-wave geometry."),
            quant="Q6_K", shape="type14-mmvq",
            files=["ggml/src/ggml-cuda/vecdotq.cuh"],
            symbols=["vec_dot_q6_K_q8_1", "vec_dot_q6_K_q8_1_impl_mmvq"],
            template="cuda-vecdotq-q6k-v1", mechanism="q6k_packed_decode_address_reuse",
            change_class="arithmetic", signatures=[q6],
            share=3.9210764977513834,
            expected_gain=(0.1, 0.6), cost="medium", risk="medium",
            falsifiers=["Any MUL_MAT correctness mismatch falsifies the candidate",
                        "Exact type-14 duration is non-positive",
                        "Exact type-14 route geometry drifts",
                        "Graphs-on throughput direction is non-positive"],
            notes=["Arithmetic lever only", "Do not change wave count", "Retain dp4a"],
            evidence_refs=[EVIDENCE, "ev-quant-ladder-occupancy-knee-20260816",
                           "ev-quant-ladder-np-raw-20260816"],
            next_action="Author one Q6-only packed-offset reuse variant under the narrow template."),
        _record(
            rank=6, hypothesis_id="akh-v26-fa-gqa7-common-map",
            title="Flash-attention GQA7 operation-level common mapping",
            statement=("One operation-level 7=3x2+1 mapping can emit exact paired-head and tail "
                       "tile routes without per-KV host slicing or axis remapping."),
            quant="Q4_K_M", shape="D64-Q1-GQA7",
            files=["ggml/src/ggml-cuda/fattn-common.cuh",
                   "ggml/src/ggml-cuda/fattn-tile.cuh"],
            symbols=["launch_fattn", "launch_fattn_tile_switch_ncols1",
                     "launch_fattn_tile_switch_ncols2"],
            template="cuda-fattn-gqa7-common-v1",
            mechanism="fattn_gqa7_operation_level_pair_tail_mapping",
            change_class="dispatcher", signatures=gqa, share=8.536375294887034,
            expected_gain=(0.1, 1.0), cost="high", risk="high",
            falsifiers=["Any generic or dedicated GQA7 correctness mismatch falsifies the candidate",
                        "Candidate routes differ from the exact 3096 pair, tail, and combine topology",
                        "Exact flash-attention duration is non-positive",
                        "Graphs-on throughput direction is non-positive"],
            notes=["Narrow two-file authority", "Preserve Q/K/V/dst axes", "No per-KV host loop"],
            evidence_refs=[EVIDENCE, "ev-source-plan-v1"],
            next_action="Author one bounded common-plus-tile mapping after exact multi-file validation."),
    ]
    body["hypotheses"] = fresh + old
    return body


def main() -> int:
    body = generate()
    hypothesis_portfolio.validate(body)
    TARGET.write_text(json.dumps(body, sort_keys=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
