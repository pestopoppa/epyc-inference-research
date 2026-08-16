"""Versioned HyRA SOL-ExecBench reference corpus for gfx90a authoring.

The upstream artifacts are NVIDIA/Hopper results, not MI210 implementations or
performance anchors.  This offline registry makes that boundary mechanical:
every seed remains reference-only until a fresh implementation is authored and
correctness/performance are re-attested on the physical gfx90a target.

This module performs no fetch, compile, profiling, inference, or scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA = "epyc.autokernel.c5_seed_corpus.v1"
CORPUS_ID = "hyra-sol-execbench-gfx90a-seeds-v1"
EXPECTED_SEED_IDS = ("k138", "k145", "k154", "k175", "k215", "k225", "k227", "k228")
DIRECT_TRITON_IDS = frozenset({"k138", "k145", "k154", "k175", "k228"})
CUDA_REAUTHOR_IDS = frozenset({"k215", "k225", "k227"})
DIRECT_TRITON = "direct_triton_spec"
CUDA_REAUTHOR = "cuda_bound_reauthor_target"
TARGET_ARCH = "gfx90a"
TARGET_HARDWARE = "AMD Instinct MI210"
DISPOSITION = "re_author_and_re_attest"
REQUIRED_MECHANISMS = ("wavefront_64", "mfma", "lds")
ALLOWED_DTYPES = frozenset({"bf16", "fp16"})
BOUND_QUALITIES = frozenset({"meaningful", "loose", "vacuous"})
TRAFFIC_BASES = frozenset({"not_flagged", "defective_declared_traffic"})
SOL_BOUND_EVIDENCE_ID = "ev-gfx90a-sol-bound-quality-20260815"
EXPECTED_BOUND_QUALITY = {
    "k138": "loose",
    "k145": "loose",
    "k154": "vacuous",
    "k175": "loose",
    "k215": "meaningful",
    "k225": "vacuous",
    "k227": "vacuous",
    "k228": "vacuous",
}
EXPECTED_HEADROOM = {
    "k138": 33.4,
    "k145": 16.8,
    "k154": 506.0,
    "k175": 17.0,
    "k215": 6.8,
    "k225": 5710.0,
    "k227": 3690.0,
    "k228": 36837.0,
}
BASE_ALLOWED_CLAIMS = ("correctness_oracle", "pytorch_relative_speed")
SOL_ALLOWED_CLAIMS = BASE_ALLOWED_CLAIMS + (
    "source_sol_score_with_bound_quality_label",
)
_SEED_RE = re.compile(r"k[0-9]{3}")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_-]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class SeedCorpusError(ValueError):
    """The checked-in corpus cannot support a deterministic authoring task."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SeedCorpusError(f"{label} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SeedCorpusError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise SeedCorpusError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _sha256(value: Any, label: str) -> str:
    value = _text(value, label)
    if not _SHA256_RE.fullmatch(value):
        raise SeedCorpusError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_float(value: Any, label: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SeedCorpusError(f"{label} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or rendered <= 0:
        raise SeedCorpusError(f"{label} must be finite and positive")
    if maximum is not None and rendered > maximum:
        raise SeedCorpusError(f"{label} must be <= {maximum}")
    return rendered


def _tuple_of_text(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SeedCorpusError(f"{label} must be a list")
    rows = tuple(_text(item, f"{label}[]") for item in value)
    if len(rows) != len(set(rows)):
        raise SeedCorpusError(f"{label} must not contain duplicates")
    return rows


@dataclass(frozen=True)
class C5Seed:
    seed_id: str
    submission_id: int
    slug: str
    artifact: str
    artifact_sha256: str
    reference_kind: str
    operator_family: str
    dtypes: tuple[str, ...]
    observed_nvidia_bindings: tuple[str, ...]
    sol_score: float
    latency_ms: float
    source_use: str
    bound_quality: str
    median_headroom_t_b_over_t_sol: float
    traffic_basis: str
    allowed_claims: tuple[str, ...]
    current_frame_correctness_eligible: bool
    current_frame_sol_score_eligible: bool
    current_frame_reason: str
    policy_evidence_ref: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "C5Seed":
        _exact_keys(row, {
            "seed_id", "submission_id", "slug", "artifact", "artifact_sha256",
            "reference_kind", "operator_family", "dtypes", "observed_nvidia_bindings",
            "upstream_result", "gfx90a", "sol_bound",
        }, "seed")
        seed_id = _text(row.get("seed_id"), "seed.seed_id")
        if not _SEED_RE.fullmatch(seed_id):
            raise SeedCorpusError(f"invalid seed id {seed_id!r}")
        if seed_id not in EXPECTED_SEED_IDS:
            raise SeedCorpusError(f"unexpected seed id {seed_id!r}")
        submission_id = row.get("submission_id")
        if (
            isinstance(submission_id, bool)
            or not isinstance(submission_id, int)
            or submission_id < 1
        ):
            raise SeedCorpusError(f"{seed_id}: submission_id must be positive")
        slug = _text(row.get("slug"), f"{seed_id}.slug")
        if not _SLUG_RE.fullmatch(slug):
            raise SeedCorpusError(f"{seed_id}: invalid slug")
        artifact = _text(row.get("artifact"), f"{seed_id}.artifact")
        parsed = PurePosixPath(artifact)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise SeedCorpusError(f"{seed_id}: artifact must be repository-relative")
        expected_prefix = f"{seed_id}__sid{submission_id}__{slug}__"
        if not parsed.name.startswith(expected_prefix) or parsed.suffix != ".json":
            raise SeedCorpusError(f"{seed_id}: artifact filename does not bind id/submission/slug")
        kind = _text(row.get("reference_kind"), f"{seed_id}.reference_kind")
        if kind not in (DIRECT_TRITON, CUDA_REAUTHOR):
            raise SeedCorpusError(f"{seed_id}: unknown reference kind {kind!r}")
        family = _text(row.get("operator_family"), f"{seed_id}.operator_family")
        if not _SLUG_RE.fullmatch(family):
            raise SeedCorpusError(f"{seed_id}: invalid operator family")
        dtypes = _tuple_of_text(row.get("dtypes"), f"{seed_id}.dtypes")
        if not dtypes or set(dtypes) - ALLOWED_DTYPES:
            raise SeedCorpusError(f"{seed_id}: dtypes must be bf16/fp16")
        bindings = _tuple_of_text(
            row.get("observed_nvidia_bindings"), f"{seed_id}.observed_nvidia_bindings"
        )
        if bindings != tuple(sorted(bindings)):
            raise SeedCorpusError(f"{seed_id}: NVIDIA bindings must be sorted")
        if kind == CUDA_REAUTHOR and not bindings:
            raise SeedCorpusError(f"{seed_id}: CUDA-bound targets must name a binding")

        result = _mapping(row.get("upstream_result"), f"{seed_id}.upstream_result")
        if result.get("status") != "COMPLETED" or result.get("is_correct") is not True:
            raise SeedCorpusError(f"{seed_id}: upstream result must be completed/correct")
        gfx90a = _mapping(row.get("gfx90a"), f"{seed_id}.gfx90a")
        if gfx90a.get("disposition") != DISPOSITION:
            raise SeedCorpusError(f"{seed_id}: gfx90a must require re-author/re-attest")
        if gfx90a.get("attestation_status") != "absent":
            raise SeedCorpusError(f"{seed_id}: no gfx90a attestation exists in this corpus")
        expected_use = (
            "reference_spec_only" if kind == DIRECT_TRITON else "behavior_and_target_only"
        )
        if gfx90a.get("source_use") != expected_use:
            raise SeedCorpusError(f"{seed_id}: source_use conflicts with reference kind")
        sol_bound = _mapping(row.get("sol_bound"), f"{seed_id}.sol_bound")
        _exact_keys(sol_bound, {
            "quality", "median_headroom_t_b_over_t_sol", "traffic_basis",
            "allowed_claims", "current_frame_eligibility", "evidence_ref",
        }, f"{seed_id}.sol_bound")
        quality = _text(sol_bound.get("quality"), f"{seed_id}.sol_bound.quality")
        if quality not in BOUND_QUALITIES or quality != EXPECTED_BOUND_QUALITY[seed_id]:
            raise SeedCorpusError(f"{seed_id}: SOL bound-quality label drifted")
        headroom = _positive_float(
            sol_bound.get("median_headroom_t_b_over_t_sol"),
            f"{seed_id}.sol_bound.median_headroom_t_b_over_t_sol",
        )
        if headroom != EXPECTED_HEADROOM[seed_id]:
            raise SeedCorpusError(f"{seed_id}: SOL bound headroom drifted")
        traffic = _text(sol_bound.get("traffic_basis"), f"{seed_id}.sol_bound.traffic_basis")
        expected_traffic = "defective_declared_traffic" if seed_id == "k175" else "not_flagged"
        if traffic not in TRAFFIC_BASES or traffic != expected_traffic:
            raise SeedCorpusError(f"{seed_id}: declared-traffic policy drifted")
        allowed_claims = _tuple_of_text(
            sol_bound.get("allowed_claims"), f"{seed_id}.sol_bound.allowed_claims"
        )
        expected_claims = (
            SOL_ALLOWED_CLAIMS
            if quality != "vacuous" and traffic == "not_flagged"
            else BASE_ALLOWED_CLAIMS
        )
        if allowed_claims != expected_claims:
            raise SeedCorpusError(f"{seed_id}: allowed SOL claims contradict bound quality")
        eligibility = _mapping(
            sol_bound.get("current_frame_eligibility"),
            f"{seed_id}.sol_bound.current_frame_eligibility",
        )
        _exact_keys(
            eligibility, {"correctness_oracle", "sol_score", "reason"},
            f"{seed_id}.sol_bound.current_frame_eligibility",
        )
        if eligibility.get("correctness_oracle") is not True:
            raise SeedCorpusError(f"{seed_id}: current gfx90a correctness oracle must stay eligible")
        if eligibility.get("sol_score") is not False:
            raise SeedCorpusError(f"{seed_id}: current gfx90a SOL score lacks measured constants")
        eligibility_reason = _text(
            eligibility.get("reason"), f"{seed_id}.sol_bound.current_frame_eligibility.reason"
        )
        evidence_ref = _text(
            sol_bound.get("evidence_ref"), f"{seed_id}.sol_bound.evidence_ref"
        )
        if evidence_ref != SOL_BOUND_EVIDENCE_ID:
            raise SeedCorpusError(f"{seed_id}: SOL policy evidence reference drifted")
        return cls(
            seed_id=seed_id,
            submission_id=submission_id,
            slug=slug,
            artifact=artifact,
            artifact_sha256=_sha256(row.get("artifact_sha256"), f"{seed_id}.artifact_sha256"),
            reference_kind=kind,
            operator_family=family,
            dtypes=dtypes,
            observed_nvidia_bindings=bindings,
            sol_score=_positive_float(result.get("sol_score"), f"{seed_id}.sol_score", maximum=1.0),
            latency_ms=_positive_float(result.get("latency_ms"), f"{seed_id}.latency_ms"),
            source_use=expected_use,
            bound_quality=quality,
            median_headroom_t_b_over_t_sol=headroom,
            traffic_basis=traffic,
            allowed_claims=allowed_claims,
            current_frame_correctness_eligible=True,
            current_frame_sol_score_eligible=False,
            current_frame_reason=eligibility_reason,
            policy_evidence_ref=evidence_ref,
        )

    def task_descriptor(self, *, revision: str) -> dict[str, Any]:
        """Return the non-numeric task surface safe to show an authoring agent."""
        return {
            "task_id": f"hyra-sol-execbench/{self.seed_id}",
            "operator_family": self.operator_family,
            "dtypes": list(self.dtypes),
            "reference": {
                "revision": revision,
                "artifact": self.artifact,
                "artifact_sha256": self.artifact_sha256,
                "kind": self.reference_kind,
                "source_use": self.source_use,
                "observed_nvidia_bindings": list(self.observed_nvidia_bindings),
            },
            "target": {
                "hardware": TARGET_HARDWARE,
                "architecture": TARGET_ARCH,
                "required_mechanisms": list(REQUIRED_MECHANISMS),
                "disposition": DISPOSITION,
                "attestation_status": "absent",
            },
            "sol_bound_policy": {
                "quality": self.bound_quality,
                "traffic_basis": self.traffic_basis,
                "allowed_claims": list(self.allowed_claims),
                "current_frame_eligibility": {
                    "correctness_oracle": self.current_frame_correctness_eligible,
                    "sol_score": self.current_frame_sol_score_eligible,
                    "reason": self.current_frame_reason,
                },
                "evidence_ref": self.policy_evidence_ref,
            },
        }


@dataclass(frozen=True)
class C5SeedCorpus:
    source_url: str
    source_revision: str
    artifact_root: str
    intake_ref: str
    policy_evidence: tuple[Mapping[str, Any], ...]
    seeds: tuple[C5Seed, ...]
    registry_sha256: str

    @classmethod
    def from_dict(cls, document: Mapping[str, Any], *, registry_sha256: str) -> "C5SeedCorpus":
        _exact_keys(document, {
            "schema", "corpus_id", "provenance", "source_attestation",
            "gfx90a_contract", "policy_evidence", "seeds",
        }, "corpus")
        if document.get("schema") != SCHEMA or document.get("corpus_id") != CORPUS_ID:
            raise SeedCorpusError("unexpected C5 seed-corpus schema or corpus id")
        provenance = _mapping(document.get("provenance"), "provenance")
        if provenance.get("project") != "Tencent-Hunyuan/Hyra-results":
            raise SeedCorpusError("provenance project must name Hyra-results")
        source_url = _text(provenance.get("url"), "provenance.url")
        if source_url != "https://github.com/Tencent-Hunyuan/Hyra-results":
            raise SeedCorpusError("provenance URL must name the upstream repository")
        revision = _text(provenance.get("revision"), "provenance.revision")
        if not _COMMIT_RE.fullmatch(revision):
            raise SeedCorpusError("provenance revision must be a full lowercase commit")
        artifact_root = _text(provenance.get("artifact_root"), "provenance.artifact_root")
        intake_ref = _text(provenance.get("intake_ref"), "provenance.intake_ref")

        attestation = _mapping(document.get("source_attestation"), "source_attestation")
        if attestation != {
            "platform_family": "nvidia_hopper",
            "scope": "nvidia_only",
            "basis": "operator_handoff_intake_884",
        }:
            raise SeedCorpusError("source attestation must stay NVIDIA/Hopper-only")
        target = _mapping(document.get("gfx90a_contract"), "gfx90a_contract")
        if (
            target.get("hardware") != TARGET_HARDWARE
            or target.get("architecture") != TARGET_ARCH
            or target.get("required_mechanisms") != list(REQUIRED_MECHANISMS)
            or target.get("disposition") != DISPOSITION
            or target.get("attestation_status") != "absent"
        ):
            raise SeedCorpusError("gfx90a contract must require fresh MI210 authoring/attestation")

        policy_evidence_raw = document.get("policy_evidence")
        if not isinstance(policy_evidence_raw, list) or len(policy_evidence_raw) != 1:
            raise SeedCorpusError("policy_evidence must contain the one SOL-bound authority")
        evidence = _mapping(policy_evidence_raw[0], "policy_evidence[0]")
        _exact_keys(evidence, {"evidence_id", "path", "sha256", "claims"}, "policy_evidence[0]")
        if evidence.get("evidence_id") != SOL_BOUND_EVIDENCE_ID:
            raise SeedCorpusError("policy evidence id drifted")
        evidence_path = _text(evidence.get("path"), "policy_evidence[0].path")
        if not Path(evidence_path).is_absolute() or ".." in Path(evidence_path).parts:
            raise SeedCorpusError("policy evidence path must be absolute and traversal-free")
        _sha256(evidence.get("sha256"), "policy_evidence[0].sha256")
        claims = _tuple_of_text(evidence.get("claims"), "policy_evidence[0].claims")
        if len(claims) < 4:
            raise SeedCorpusError("policy evidence must carry the complete bound-quality claim set")

        rows = document.get("seeds")
        if not isinstance(rows, list):
            raise SeedCorpusError("seeds must be a list")
        seeds = tuple(C5Seed.from_dict(_mapping(row, "seed")) for row in rows)
        ids = tuple(seed.seed_id for seed in seeds)
        if ids != EXPECTED_SEED_IDS:
            raise SeedCorpusError(f"seed ids/order must be {EXPECTED_SEED_IDS}")
        direct = {seed.seed_id for seed in seeds if seed.reference_kind == DIRECT_TRITON}
        cuda = {seed.seed_id for seed in seeds if seed.reference_kind == CUDA_REAUTHOR}
        if direct != DIRECT_TRITON_IDS or cuda != CUDA_REAUTHOR_IDS:
            raise SeedCorpusError("direct/CUDA-bound seed partition drifted")
        prefix = artifact_root.rstrip("/") + "/"
        if any(not seed.artifact.startswith(prefix) for seed in seeds):
            raise SeedCorpusError("a seed artifact escapes the declared artifact root")
        if len({seed.artifact for seed in seeds}) != len(seeds):
            raise SeedCorpusError("seed artifacts must be unique")
        return cls(
            source_url=source_url,
            source_revision=revision,
            artifact_root=artifact_root,
            intake_ref=intake_ref,
            policy_evidence=(dict(evidence),),
            seeds=seeds,
            registry_sha256=_sha256(registry_sha256, "registry_sha256"),
        )

    def select(self, seed_ids: Sequence[str] | None = None) -> tuple[C5Seed, ...]:
        if seed_ids is None:
            return self.seeds
        requested = tuple(seed_ids)
        if not requested or len(requested) != len(set(requested)):
            raise SeedCorpusError("seed selection must be non-empty and unique")
        by_id = {seed.seed_id: seed for seed in self.seeds}
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise SeedCorpusError(f"unknown C5 seed ids: {unknown}")
        return tuple(by_id[seed_id] for seed_id in requested)


def _registry_path() -> Path:
    return Path(__file__).with_name("c5_seed_corpus.json")


def load(path: str | Path | None = None) -> C5SeedCorpus:
    registry = _registry_path() if path is None else Path(path)
    raw = registry.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SeedCorpusError("C5 seed corpus is not valid JSON") from exc
    return C5SeedCorpus.from_dict(
        _mapping(document, "corpus"),
        registry_sha256=hashlib.sha256(raw).hexdigest(),
    )


def seed_context_item(seed_ids: Sequence[str] | None = None):
    """Project selected reference tasks into the priced authoring-context seam."""
    from .controller.authoring_contract import ContextItem, assert_prompt_hygiene

    corpus = load()
    selected = corpus.select(seed_ids)
    payload = {
        "schema": SCHEMA,
        "authority": "reference_only",
        "source": {
            "url": corpus.source_url,
            "revision": corpus.source_revision,
            "scope": "nvidia_hopper_only",
        },
        "policy_evidence": [
            {key: evidence[key] for key in ("evidence_id", "path", "sha256")}
            for evidence in corpus.policy_evidence
        ],
        "tasks": [seed.task_descriptor(revision=corpus.source_revision) for seed in selected],
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert_prompt_hygiene(content)
    selection_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return ContextItem(
        source_ref=f"hyra-c5://{corpus.registry_sha256}/{selection_sha}",
        purpose="HyRA SOL-ExecBench reference tasks requiring gfx90a re-authoring",
        content=content,
    )
