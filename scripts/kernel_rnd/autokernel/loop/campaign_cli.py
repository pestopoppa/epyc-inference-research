"""Offline campaign resolver; this module has no launch or resource-claim path."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

from . import campaign
from .status import write_json


REGISTRY_SNAPSHOT_SCHEMA = "epyc.autokernel.artifact_registry_snapshot.v1"
DRY_RESOLUTION_SCHEMA = "epyc.autokernel.campaign_dry_resolution.v1"
ARTIFACT_KINDS = frozenset({"source", "model", "build", "recipe"})


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise campaign.ManifestError(f"{label} must be an object")
    return dict(value)


def _exact_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unknown {sorted(extra)}")
        raise campaign.ManifestError(f"{label}: " + "; ".join(details))


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise campaign.ManifestError(f"cannot load {label} {path}: {exc}") from exc


def load_registry_snapshot(path: Path | str) -> dict[str, dict[str, dict[str, str]]]:
    """Read and strictly validate one caller-captured artifact registry snapshot."""
    row = _object(_read_json(Path(path), "registry snapshot"), "registry snapshot")
    _exact_keys(row, {"schema", "artifacts"}, "registry snapshot")
    if row["schema"] != REGISTRY_SNAPSHOT_SCHEMA:
        raise campaign.ManifestError(
            f"registry snapshot: unsupported schema {row['schema']!r}")
    groups = _object(row["artifacts"], "registry snapshot.artifacts")
    unknown = set(groups) - ARTIFACT_KINDS
    if unknown:
        raise campaign.ManifestError(
            f"registry snapshot.artifacts: unknown kinds {sorted(unknown)}")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for kind, value in groups.items():
        entries = _object(value, f"registry snapshot.artifacts.{kind}")
        result[kind] = {}
        for ref, identity in entries.items():
            if not isinstance(ref, str) or not ref.strip():
                raise campaign.ManifestError(
                    f"registry snapshot.artifacts.{kind} refs must be non-empty strings")
            result[kind][ref] = campaign.ArtifactIdentity.from_dict(
                identity, kind=kind, ref=ref).to_dict()
    return result


def load_previous(path: Path | str) -> campaign.ResolvedCampaign:
    """Load either a raw resolved campaign or this CLI's versioned output envelope."""
    row = _object(_read_json(Path(path), "previous resolution"), "previous resolution")
    if row.get("schema") == campaign.RESOLVED_SCHEMA:
        return campaign.ResolvedCampaign.from_dict(row)
    if row.get("schema") != DRY_RESOLUTION_SCHEMA:
        raise campaign.ManifestError(
            f"previous resolution: unsupported schema {row.get('schema')!r}")
    _exact_keys(row, {"schema", "mode", "admission_ready", "disposition",
                      "resolved_campaign", "target_dispositions", "summary", "verification"},
                "previous resolution")
    return campaign.ResolvedCampaign.from_dict(row["resolved_campaign"])


def _artifact_uses(resolved: campaign.ResolvedCampaign
                   ) -> dict[tuple[str, str, str, str], tuple[campaign.ArtifactIdentity, set[str]]]:
    uses: dict[tuple[str, str, str, str], tuple[campaign.ArtifactIdentity, set[str]]] = {}
    all_targets = {target_id for target in resolved.targets for target_id in target.target_ids}

    def add(identity: campaign.ArtifactIdentity | None, target_ids: set[str]) -> None:
        if identity is None:
            return
        key = (identity.kind, identity.ref, identity.path, identity.sha256)
        if key not in uses:
            uses[key] = (identity, set())
        uses[key][1].update(target_ids)

    for _, identity in resolved.source_snapshot:
        add(identity, all_targets)
    for target in resolved.targets:
        target_ids = set(target.target_ids)
        for identity in (target.execution.model, target.execution.build,
                         target.execution.recipe, target.execution.drafter,
                         target.baseline):
            add(identity, target_ids)
    return uses


def _verify_identity(identity: campaign.ArtifactIdentity) -> dict[str, Any] | None:
    path = Path(identity.path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"reason": "missing", "observed_sha256": None}
    except OSError as exc:
        return {"reason": f"unreadable:{exc}", "observed_sha256": None}
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        return {"reason": "not_regular_file", "observed_sha256": None}
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return {"reason": f"unreadable:{exc}", "observed_sha256": None}
    observed = digest.hexdigest()
    if observed != identity.sha256:
        return {"reason": "sha256_mismatch", "observed_sha256": observed}
    return None


def build_output(resolved: campaign.ResolvedCampaign, *, verify_artifacts: bool
                 ) -> dict[str, Any]:
    """Project a resolution and optional local-file checks without changing it."""
    errors: list[dict[str, Any]] = []
    uses = _artifact_uses(resolved)
    if verify_artifacts:
        for _, (identity, target_ids) in sorted(uses.items()):
            failure = _verify_identity(identity)
            if failure is not None:
                errors.append({"kind": identity.kind, "ref": identity.ref,
                               "path": identity.path, "expected_sha256": identity.sha256,
                               "target_ids": sorted(target_ids), **failure})

    dispositions = []
    for target in resolved.targets:
        relevant = [error for error in errors
                    if set(target.target_ids) & set(error["target_ids"])]
        dispositions.append({
            "target_ids": list(target.target_ids),
            "resolution_status": target.status,
            "verification_status": (
                "failed" if relevant else "passed" if verify_artifacts else "not_requested"),
            "verification_errors": [
                {key: value for key, value in error.items() if key != "target_ids"}
                for error in relevant],
        })

    resolution_counts = Counter(item["resolution_status"] for item in dispositions)
    verification_counts = Counter(item["verification_status"] for item in dispositions)
    partial = any(item.status != "ready" for item in resolved.targets) or bool(errors)
    return {
        "schema": DRY_RESOLUTION_SCHEMA,
        "mode": "offline_dry_resolution",
        # Resolution and even file verification grant no compute or admission authority.
        "admission_ready": False,
        "disposition": ("partial" if partial else
                        "verified_resolution" if verify_artifacts else
                        "resolved_unverified"),
        "resolved_campaign": resolved.to_dict(),
        "target_dispositions": dispositions,
        "summary": {
            "targets": len(dispositions),
            "resolution_dispositions": dict(sorted(resolution_counts.items())),
            "verification_dispositions": dict(sorted(verification_counts.items())),
        },
        "verification": {
            "requested": verify_artifacts,
            "status": ("failed" if errors else "passed" if verify_artifacts
                       else "not_requested"),
            "checked_artifacts": len(uses) if verify_artifacts else 0,
            "errors": errors,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a unified AutoKernel campaign offline; never execute it")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registry-snapshot", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-artifacts", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = campaign.load_manifest(args.manifest)
        registry = load_registry_snapshot(args.registry_snapshot)
        previous = load_previous(args.previous) if args.previous is not None else None
        resolved = campaign.resolve_manifest(
            manifest, registry_snapshot=registry, previous=previous)
        output = build_output(resolved, verify_artifacts=args.verify_artifacts)
        if args.out is None:
            json.dump(output, sys.stdout, indent=2, sort_keys=True)
            sys.stdout.write("\n")
        else:
            write_json(args.out.parent, args.out.name, output, prefix=".campaign-")
    except (campaign.ManifestError, OSError) as exc:
        print(f"campaign dry resolution refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DRY_RESOLUTION_SCHEMA", "REGISTRY_SNAPSHOT_SCHEMA", "build_output",
           "load_previous", "load_registry_snapshot", "main"]
