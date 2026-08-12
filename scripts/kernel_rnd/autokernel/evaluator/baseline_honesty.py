"""Exact-surface baseline selection for AutoKernel evaluator cells."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .. import schemas
from ..execution import provider as provider_isolation


VENDOR_PREFILL_PROVIDERS = frozenset(("rocblas", "hipblaslt"))
EXPLICIT_FACTOR_NAMES = frozenset(("flash_attention", "mmq_mfma", "rocwmma_fattn"))
PROVIDER_MANIFEST_SCHEMA = "epyc.autokernel.baseline_provider_manifest.v1"


@dataclass(frozen=True)
class ProviderManifest:
    """Exact library identity; a provider label is not provenance."""

    schema: str
    provider: str
    package_version: str
    library_binary_sha256: str
    reference: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema != PROVIDER_MANIFEST_SCHEMA:
            raise ValueError(f"provider manifest schema must be {PROVIDER_MANIFEST_SCHEMA!r}")
        if self.provider not in VENDOR_PREFILL_PROVIDERS:
            raise ValueError(f"unsupported provider manifest: {self.provider!r}")
        if not isinstance(self.package_version, str) or not self.package_version.strip():
            raise ValueError("provider package_version is required")
        if not schemas.SHA256_RE.fullmatch(self.library_binary_sha256) \
                or schemas.is_placeholder_digest(self.library_binary_sha256):
            raise ValueError("provider library_binary_sha256 must be a real SHA-256")
        violations = schemas.validate_provider_reference(
            self.reference, "provider_manifest.reference.")
        if violations:
            raise ValueError("invalid provider reference: " + "; ".join(violations))
        try:
            provider_isolation.IsolatedProviderPrefix.create(
                str(self.reference["isolation_root"]))
        except provider_isolation.ProviderIsolationError as exc:
            raise ValueError(f"invalid provider isolation: {exc}") from exc
        if self.reference.get("kind") != "rocm_library" \
                or self.reference.get("target_backend") != "llama_gpu" \
                or self.reference.get("evidence_authority") != "diagnostic_only":
            raise ValueError(
                "baseline provider reference must be a diagnostic-only llama_gpu "
                "ROCm library")

    def to_dict(self) -> dict:
        return {
            "schema": self.schema, "provider": self.provider,
            "package_version": self.package_version,
            "library_binary_sha256": self.library_binary_sha256,
            "reference": dict(self.reference),
        }


@dataclass(frozen=True)
class SurfaceKey:
    workload: str
    backend: str
    model_sha256: str
    quant: str
    operation: str
    shape: tuple[int, ...]
    dtype: str
    build_sha256: str
    factors: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, *, workload: str, backend: str, model_sha256: str, quant: str,
               operation: str, shape: Sequence[int], dtype: str, build_sha256: str,
               factors: Mapping[str, object]) -> "SurfaceKey":
        text_fields = {
            "workload": workload,
            "backend": backend,
            "model_sha256": model_sha256,
            "quant": quant,
            "operation": operation,
            "dtype": dtype,
            "build_sha256": build_sha256,
        }
        for name, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"surface {name} must be a non-empty string")
        normalized_shape = tuple(shape)
        if not normalized_shape or any(isinstance(value, bool) or not isinstance(value, int)
                                       or value <= 0 for value in normalized_shape):
            raise ValueError("surface shape must contain positive integers")
        missing = EXPLICIT_FACTOR_NAMES - set(factors)
        if missing:
            raise ValueError(f"surface leaves evaluator factors implicit: {sorted(missing)}")
        normalized_factors = tuple(sorted((str(key), str(value))
                                          for key, value in factors.items()))
        if any(value.lower() == "auto" for _key, value in normalized_factors):
            raise ValueError("surface factors must be resolved; 'auto' is not an identity")
        return cls(shape=normalized_shape, factors=normalized_factors, **text_fields)


@dataclass(frozen=True)
class BaselineObservation:
    provider: str
    surface: SurfaceKey
    metric: float
    metric_id: str
    evidence_ref: str
    provider_manifest: ProviderManifest

    def __post_init__(self) -> None:
        if self.provider not in VENDOR_PREFILL_PROVIDERS:
            raise ValueError(f"unsupported vendor baseline provider: {self.provider!r}")
        if isinstance(self.metric, bool) or not isinstance(self.metric, (int, float)):
            raise TypeError("baseline metric must be numeric")
        if not self.metric_id or not self.evidence_ref:
            raise ValueError("baseline metric_id and evidence_ref are required")
        if not isinstance(self.provider_manifest, ProviderManifest):
            raise TypeError("provider_manifest is required")
        if self.provider_manifest.provider != self.provider:
            raise ValueError("observation provider differs from provider manifest")


@dataclass(frozen=True)
class BaselineSelection:
    surface: SurfaceKey
    selected: BaselineObservation
    compared: tuple[BaselineObservation, ...]
    metric_direction: str

    def to_dict(self) -> dict:
        return {
            "surface": {
                **{name: getattr(self.surface, name) for name in (
                    "workload", "backend", "model_sha256", "quant", "operation",
                    "dtype", "build_sha256")},
                "shape": list(self.surface.shape),
                "factors": dict(self.surface.factors),
            },
            "selected_provider": self.selected.provider,
            "selected_metric": self.selected.metric,
            "metric_id": self.selected.metric_id,
            "metric_direction": self.metric_direction,
            "evidence_ref": self.selected.evidence_ref,
            "selected_provider_manifest": self.selected.provider_manifest.to_dict(),
            "compared_providers": [item.provider for item in self.compared],
            "compared_provider_manifests": [
                item.provider_manifest.to_dict() for item in self.compared],
        }


def select_strongest_prefill_baseline(
        surface: SurfaceKey, observations: Sequence[BaselineObservation], *,
        metric_direction: str = "higher_better") -> BaselineSelection:
    """Select the stronger vendor baseline only after exact surface matching."""
    if metric_direction not in ("higher_better", "lower_better"):
        raise ValueError(f"unsupported metric direction: {metric_direction!r}")
    matched = tuple(item for item in observations if item.surface == surface)
    providers = {item.provider for item in matched}
    if providers != VENDOR_PREFILL_PROVIDERS:
        missing = sorted(VENDOR_PREFILL_PROVIDERS - providers)
        extras = sorted(providers - VENDOR_PREFILL_PROVIDERS)
        raise ValueError(
            f"exact surface requires one rocBLAS and one hipBLASLt baseline; "
            f"missing={missing}, extras={extras}")
    if len(matched) != len(VENDOR_PREFILL_PROVIDERS):
        raise ValueError("exact surface has duplicate vendor baseline observations")
    metric_ids = {item.metric_id for item in matched}
    if len(metric_ids) != 1:
        raise ValueError("vendor baselines use different metrics")
    def metric_value(item: BaselineObservation) -> float:
        return item.metric

    selected = (max if metric_direction == "higher_better" else min)(
        matched, key=metric_value)
    return BaselineSelection(
        surface=surface, selected=selected,
        compared=tuple(sorted(matched, key=lambda item: item.provider)),
        metric_direction=metric_direction)


def require_candidate_surface(selection: BaselineSelection, candidate: SurfaceKey) -> None:
    """Refuse model, quant, shape, build, or factor transfer from a selected baseline."""
    if candidate != selection.surface:
        raise ValueError("candidate surface differs from the measured baseline surface")
