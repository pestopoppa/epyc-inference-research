"""Replayable enrollment requests and explicit resolution of missing artifacts.

The registry is a caller-supplied snapshot, never an allocation or launch grant.
Existing registry entries are immutable within a request; only missing entries may
be added by a resolution event. Controller ownership serializes these operations.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any, Mapping

from . import campaign

SCHEMA = "epyc.autokernel.enrollment_event.v1"


class EnrollmentRefused(ValueError):
    pass


def validate_event(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != {
            "schema", "event", "campaign_id", "manifest", "registry", "resolved"}:
        raise EnrollmentRefused("enrollment event fields differ")
    if (record["schema"] != SCHEMA or not isinstance(record["event"], str)
            or record["event"] not in {"ENROLLED", "ARTIFACTS_RESOLVED"}
            or not isinstance(record["campaign_id"], str) or not record["campaign_id"]
            or not isinstance(record["registry"], Mapping)):
        raise EnrollmentRefused("enrollment event envelope differs")
    try:
        encoded = json.dumps(record, sort_keys=True, allow_nan=False)
        if json.loads(encoded) != record:
            raise EnrollmentRefused("enrollment event is not canonical JSON data")
    except (TypeError, ValueError) as exc:
        raise EnrollmentRefused("enrollment event is not canonical JSON data") from exc
    try:
        manifest = campaign.CampaignManifest.from_dict(record["manifest"])
        resolved = campaign.ResolvedCampaign.from_dict(record["resolved"])
    except (campaign.ManifestError, TypeError, KeyError) as exc:
        raise EnrollmentRefused(str(exc)) from exc
    if (manifest.campaign_id != record["campaign_id"] or
            resolved.campaign_id != manifest.campaign_id or
            resolved.request_id != manifest.request_id or
            resolved.manifest_digest != manifest.manifest_digest):
        raise EnrollmentRefused("enrollment request identity differs")


def project(records: list[Mapping[str, Any]], campaign_id: str,
            initial: campaign.ResolvedCampaign) -> dict[str, dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    previous = initial
    for index, record in enumerate(records):
        validate_event(record)
        if record["campaign_id"] != campaign_id:
            raise EnrollmentRefused("enrollment campaign differs")
        try:
            manifest = campaign.CampaignManifest.from_dict(record["manifest"])
            registry = record["registry"]
            if manifest.campaign_id != campaign_id or not isinstance(registry, Mapping):
                raise EnrollmentRefused("invalid campaign or registry")
            request = manifest.request_id
            old = requests.get(request)
            if index == 0 and request != initial.request_id:
                raise EnrollmentRefused("initial campaign request must be enrolled first")
            if record["event"] == "ENROLLED":
                if old is not None:
                    raise EnrollmentRefused("duplicate enrollment event")
                if request == initial.request_id and manifest.manifest_digest != initial.manifest_digest:
                    raise EnrollmentRefused("initial request identity differs")
                if index == 0 and request == initial.request_id:
                    fresh = campaign.resolve_manifest(manifest, registry_snapshot=registry)
                    expected = initial.to_dict()
                    actual = fresh.to_dict()
                    for target in expected["targets"]:
                        target["revision"] = 1
                    if actual != expected:
                        raise EnrollmentRefused("initial registry does not match frozen campaign")
                result = campaign.resolve_manifest(manifest, registry_snapshot=registry,
                                                   previous=previous)
            elif record["event"] == "ARTIFACTS_RESOLVED":
                if old is None or old["manifest"] != manifest.to_dict():
                    raise EnrollmentRefused("resolution lacks its original request")
                # Resolve the request's own frozen projection, even when another
                # request was enrolled in the meantime.
                previous = old["resolved"]
                for kind, group in old["registry"].items():
                    if not isinstance(group, Mapping) or not isinstance(registry.get(kind), Mapping):
                        raise EnrollmentRefused("registry group disappeared")
                    for ref, identity in group.items():
                        if identity is not None and registry[kind].get(ref) != identity:
                            raise EnrollmentRefused("previous registry identity moved")
                if registry == old["registry"]:
                    raise EnrollmentRefused("resolution did not add an artifact")
                # Re-resolve under the same frozen request, while asking the pure
                # resolver to preserve the previous revisions and baseline pins.
                prior = replace(previous, request_id=previous.request_id + ":resolution-prior")
                result = campaign.resolve_manifest(manifest, registry_snapshot=registry,
                                                   previous=prior)
                if not any(any(reason.endswith(":missing") for reason in t.missing)
                           for t in previous.targets):
                    raise EnrollmentRefused("request has no missing artifact")
                old_sources = dict(previous.source_snapshot)
                new_sources = dict(result.source_snapshot)
                if any(pin is not None and new_sources.get(name) != pin
                       for name, pin in old_sources.items()):
                    raise EnrollmentRefused("source identity moved")
                old_by_id = {name: target for target in previous.targets
                             for name in target.target_ids}
                new_by_id = {name: target for target in result.targets
                             for name in target.target_ids}
                if set(old_by_id) != set(new_by_id):
                    raise EnrollmentRefused("resolution changed target membership")
                progress = any(pin is None and new_sources.get(name) is not None
                               for name, pin in old_sources.items())
                for name, old_target in old_by_id.items():
                    new_target = new_by_id[name]
                    if not any(reason.endswith(":missing") for reason in old_target.missing) and (
                            old_target.execution != new_target.execution
                            or old_target.baseline != new_target.baseline
                            or old_target.status != new_target.status):
                        raise EnrollmentRefused("resolved target identity moved")
                    for field in ("model", "build", "recipe", "drafter"):
                        old_pin = getattr(old_target.execution, field)
                        new_pin = getattr(new_target.execution, field)
                        if old_pin is not None and old_pin != new_pin:
                            raise EnrollmentRefused("resolved artifact identity moved")
                        progress |= old_pin is None and new_pin is not None
                    if old_target.baseline is not None and old_target.baseline != new_target.baseline:
                        raise EnrollmentRefused("baseline identity moved")
                    progress |= old_target.baseline is None and new_target.baseline is not None
                if not progress:
                    raise EnrollmentRefused("resolution did not resolve a missing artifact")
            else:
                raise EnrollmentRefused("unknown enrollment event")
            if result.to_dict() != record["resolved"]:
                raise EnrollmentRefused("resolved projection differs")
            requests[request] = {"manifest": manifest.to_dict(), "event": record["event"],
                                 "registry": copy.deepcopy(dict(registry)), "resolved": result}
            previous = result
        except (campaign.ManifestError, TypeError, KeyError, AttributeError) as exc:
            raise EnrollmentRefused(str(exc)) from exc
    return requests


def make_event(event: str, manifest: campaign.CampaignManifest, registry: Mapping[str, Any],
               records: list[Mapping[str, Any]],
               initial: campaign.ResolvedCampaign) -> dict[str, Any]:
    if not isinstance(registry, Mapping):
        raise EnrollmentRefused("registry must be a snapshot mapping")
    requests = project(records, manifest.campaign_id, initial)
    previous = (requests[records[-1]["manifest"]["request_id"]]["resolved"]
                if records else initial)
    if event == "ARTIFACTS_RESOLVED" and previous is not None:
        if manifest.request_id in requests:
            previous = requests[manifest.request_id]["resolved"]
        previous = replace(previous, request_id=previous.request_id + ":resolution-prior")
    result = campaign.resolve_manifest(manifest, registry_snapshot=registry, previous=previous)
    row = {"schema": SCHEMA, "event": event, "campaign_id": manifest.campaign_id,
           "manifest": manifest.to_dict(), "registry": copy.deepcopy(dict(registry)),
           "resolved": result.to_dict()}
    project([*records, row], manifest.campaign_id, initial)
    return row
