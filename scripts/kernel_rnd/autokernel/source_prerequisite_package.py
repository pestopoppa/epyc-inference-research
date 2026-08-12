"""Immutable archive/resume packages for source-candidate T0 prerequisites.

The sensitivity, hostile-distribution and checker-isolation runners write raw
CSV evidence.  A reduced PASS is not authority by itself: this module reloads
the exact content-addressed bytes, reruns the trusted reducers, and only then
binds their results to the live candidate source and evaluator identities.

Packages embed their evidence.  They contain no mutable filesystem references,
and the loader snapshots one small regular file before a campaign can acquire a
host claim.  Fresh-candidate production remains a separate execution adapter;
this module is deliberately the archive/resume seam only.
"""
from __future__ import annotations

import base64
import binascii
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from . import schemas
from .evaluator import (correctness, oracle_integrity, sensitivity,
                        source_candidate_authority)


SCHEMA = "epyc.autokernel.source-prerequisite-package.v1"
REQUIRED_IDS = frozenset(correctness.SOURCE_PREREQUISITE_IDS)
MAX_PACKAGE_BYTES = 32 * 1024 * 1024
_PACKAGE_FIELDS = frozenset((
    "schema", "campaign_id", "proposal_id", "candidate_id",
    "candidate_source_sha256", "candidate_binary_sha256",
    "evaluator_bundle_sha256", "producer_id", "capture_mode", "receipts",
    "package_sha256",
))
_RECEIPT_FIELDS = frozenset((
    "prerequisite_id", "suite_version", "documents", "receipt_sha256",
))
_DOCUMENT_FIELDS = frozenset((
    "suite_seed", "csv_sha256", "csv_base64",
))


class SourcePrerequisitePackageError(RuntimeError):
    """The package cannot safely supply source-candidate T0 evidence."""


def _require_exact_fields(value: Any, fields: frozenset[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != fields:
        got = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise SourcePrerequisitePackageError(
            f"{label} fields must be exactly {sorted(fields)}; got {got}")
    return value


def _require_id(value: Any, prefix: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith(prefix):
        raise SourcePrerequisitePackageError(f"{label} must start with {prefix!r}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    try:
        schemas.require.sha256(value, label, error=SourcePrerequisitePackageError)
    except TypeError as exc:
        raise SourcePrerequisitePackageError(str(exc)) from exc
    return value


def _canonical_without(value: Mapping[str, Any], key: str) -> dict:
    return {name: value[name] for name in sorted(value) if name != key}


def receipt_sha256(payload: Mapping[str, Any]) -> str:
    """Content identity of one receipt, including every embedded CSV byte."""
    return schemas.content_hash(_canonical_without(payload, "receipt_sha256"))


def package_sha256(payload: Mapping[str, Any]) -> str:
    """Content identity of the complete package, excluding only its own hash."""
    return schemas.content_hash(_canonical_without(payload, "package_sha256"))


@dataclass(frozen=True)
class CsvDocument:
    suite_seed: int
    csv_sha256: str
    csv_bytes: bytes

    def to_mapping(self) -> dict[str, Any]:
        return {
            "suite_seed": self.suite_seed,
            "csv_sha256": self.csv_sha256,
            "csv_base64": base64.b64encode(self.csv_bytes).decode("ascii"),
        }

    @classmethod
    def from_mapping(cls, raw: Any, *, label: str) -> "CsvDocument":
        value = _require_exact_fields(raw, _DOCUMENT_FIELDS, label)
        seed = value["suite_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise SourcePrerequisitePackageError(
                f"{label}.suite_seed must be a non-negative integer")
        expected = _require_sha256(value["csv_sha256"], f"{label}.csv_sha256")
        encoded = value["csv_base64"]
        if not isinstance(encoded, str):
            raise SourcePrerequisitePackageError(f"{label}.csv_base64 must be a string")
        try:
            body = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SourcePrerequisitePackageError(
                f"{label}.csv_base64 is not strict base64: {exc}") from exc
        if hashlib.sha256(body).hexdigest() != expected:
            raise SourcePrerequisitePackageError(
                f"{label}.csv_sha256 does not match the embedded CSV bytes")
        return cls(suite_seed=seed, csv_sha256=expected, csv_bytes=body)

    def rows(self, *, label: str) -> tuple[dict[str, str], ...]:
        try:
            text = self.csv_bytes.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise SourcePrerequisitePackageError(
                f"{label} is not strict UTF-8 CSV: {exc}") from exc
        try:
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if not reader.fieldnames or any(not name for name in reader.fieldnames):
                raise SourcePrerequisitePackageError(
                    f"{label} has no complete CSV header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise SourcePrerequisitePackageError(
                    f"{label} has duplicate CSV columns")
            rows = []
            for index, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise SourcePrerequisitePackageError(
                        f"{label} row {index} does not match its CSV header")
                rows.append(dict(row))
        except csv.Error as exc:
            raise SourcePrerequisitePackageError(
                f"{label} is malformed CSV: {exc}") from exc
        if not rows:
            raise SourcePrerequisitePackageError(f"{label} contains no evidence rows")
        return tuple(rows)


@dataclass(frozen=True)
class PrerequisiteReceipt:
    prerequisite_id: str
    suite_version: str
    documents: tuple[CsvDocument, ...]
    receipt_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "prerequisite_id": self.prerequisite_id,
            "suite_version": self.suite_version,
            "documents": [document.to_mapping() for document in self.documents],
            "receipt_sha256": self.receipt_sha256,
        }

    @classmethod
    def from_mapping(cls, raw: Any, *, index: int) -> "PrerequisiteReceipt":
        label = f"receipts[{index}]"
        value = _require_exact_fields(raw, _RECEIPT_FIELDS, label)
        prerequisite_id = value["prerequisite_id"]
        if prerequisite_id not in REQUIRED_IDS:
            raise SourcePrerequisitePackageError(
                f"{label}.prerequisite_id {prerequisite_id!r} is not one of "
                f"{sorted(REQUIRED_IDS)}")
        suite_version = value["suite_version"]
        if not isinstance(suite_version, str) or not suite_version.strip():
            raise SourcePrerequisitePackageError(
                f"{label}.suite_version must be a non-empty string")
        documents_raw = value["documents"]
        if not isinstance(documents_raw, list) or not documents_raw:
            raise SourcePrerequisitePackageError(f"{label}.documents must be non-empty")
        documents = tuple(CsvDocument.from_mapping(
            item, label=f"{label}.documents[{doc_index}]")
            for doc_index, item in enumerate(documents_raw))
        seeds = tuple(document.suite_seed for document in documents)
        if len(seeds) != len(set(seeds)):
            raise SourcePrerequisitePackageError(
                f"{label}.documents contains duplicate suite seeds")
        if prerequisite_id == "input_sensitivity" and len(documents) < 3:
            raise SourcePrerequisitePackageError(
                "input_sensitivity requires at least three distinct seed documents")
        if prerequisite_id != "input_sensitivity" and len(documents) != 1:
            raise SourcePrerequisitePackageError(
                f"{prerequisite_id} requires exactly one CSV document")
        expected = _require_sha256(value["receipt_sha256"], f"{label}.receipt_sha256")
        if receipt_sha256(value) != expected:
            raise SourcePrerequisitePackageError(
                f"{label}.receipt_sha256 does not match its embedded content")
        return cls(prerequisite_id=prerequisite_id, suite_version=suite_version,
                   documents=documents, receipt_sha256=expected)


@dataclass(frozen=True)
class SourcePrerequisitePackage:
    campaign_id: str
    proposal_id: str
    candidate_id: str
    candidate_source_sha256: str
    candidate_binary_sha256: str
    evaluator_bundle_sha256: str
    producer_id: str
    capture_mode: str
    receipts: tuple[PrerequisiteReceipt, ...]
    package_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "campaign_id": self.campaign_id,
            "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "candidate_source_sha256": self.candidate_source_sha256,
            "candidate_binary_sha256": self.candidate_binary_sha256,
            "evaluator_bundle_sha256": self.evaluator_bundle_sha256,
            "producer_id": self.producer_id,
            "capture_mode": self.capture_mode,
            "receipts": [receipt.to_mapping() for receipt in self.receipts],
            "package_sha256": self.package_sha256,
        }

    @classmethod
    def from_mapping(cls, raw: Any) -> "SourcePrerequisitePackage":
        value = _require_exact_fields(raw, _PACKAGE_FIELDS, "package")
        if value["schema"] != SCHEMA:
            raise SourcePrerequisitePackageError(
                f"package.schema must be {SCHEMA!r}")
        campaign_id = _require_id(value["campaign_id"], "ak-", "campaign_id")
        proposal_id = _require_id(value["proposal_id"], "akp-", "proposal_id")
        candidate_id = _require_id(value["candidate_id"], "akc-", "candidate_id")
        source_sha = _require_sha256(
            value["candidate_source_sha256"], "candidate_source_sha256")
        binary_sha = _require_sha256(
            value["candidate_binary_sha256"], "candidate_binary_sha256")
        evaluator_sha = _require_sha256(
            value["evaluator_bundle_sha256"], "evaluator_bundle_sha256")
        if value["producer_id"] != sensitivity.TRUSTED_PRODUCER:
            raise SourcePrerequisitePackageError(
                f"producer_id must be {sensitivity.TRUSTED_PRODUCER!r}")
        if value["capture_mode"] not in ("measured", "dry_run"):
            raise SourcePrerequisitePackageError(
                "capture_mode must be 'measured' or 'dry_run'")
        receipts_raw = value["receipts"]
        if not isinstance(receipts_raw, list):
            raise SourcePrerequisitePackageError("receipts must be a list")
        receipts = tuple(PrerequisiteReceipt.from_mapping(item, index=index)
                         for index, item in enumerate(receipts_raw))
        ids = tuple(receipt.prerequisite_id for receipt in receipts)
        if len(ids) != len(set(ids)):
            raise SourcePrerequisitePackageError("receipts contains duplicate prerequisite ids")
        if set(ids) != REQUIRED_IDS:
            raise SourcePrerequisitePackageError(
                f"receipts must contain exactly {sorted(REQUIRED_IDS)}; got {sorted(ids)}")
        versions = {receipt.suite_version for receipt in receipts}
        if len(versions) != 1:
            raise SourcePrerequisitePackageError(
                "all three prerequisite receipts must name one suite version")
        expected = _require_sha256(value["package_sha256"], "package_sha256")
        if package_sha256(value) != expected:
            raise SourcePrerequisitePackageError(
                "package_sha256 does not match the complete embedded package")
        return cls(
            campaign_id=campaign_id, proposal_id=proposal_id, candidate_id=candidate_id,
            candidate_source_sha256=source_sha, candidate_binary_sha256=binary_sha,
            evaluator_bundle_sha256=evaluator_sha, producer_id=value["producer_id"],
            capture_mode=value["capture_mode"], receipts=receipts,
            package_sha256=expected)

    def bind_campaign(self, *, proposal: Mapping[str, Any], campaign_id: str,
                      candidate_id: str) -> None:
        expected = (campaign_id, proposal.get("proposal_id"), candidate_id)
        got = (self.campaign_id, self.proposal_id, self.candidate_id)
        if got != expected:
            raise SourcePrerequisitePackageError(
                f"source prerequisite package identity {got!r} != campaign {expected!r}")
        if proposal.get("change_class") == "parameter":
            raise SourcePrerequisitePackageError(
                "parameter proposals may not carry source prerequisites")

    def materialize(self, *, candidate_source_sha256: str,
                    candidate_binary_sha256: str,
                    evaluator_bundle_sha256: str
                    ) -> tuple[correctness.SourcePrerequisiteEvidence, ...]:
        """Recompute all reducer verdicts and bind them to this live build."""
        identities = (
            ("candidate source", self.candidate_source_sha256,
             candidate_source_sha256),
            ("candidate test-backend-ops binary", self.candidate_binary_sha256,
             candidate_binary_sha256),
            ("evaluator bundle", self.evaluator_bundle_sha256,
             evaluator_bundle_sha256),
        )
        for label, declared, observed in identities:
            _require_sha256(observed, f"observed {label}")
            if declared != observed:
                raise SourcePrerequisitePackageError(
                    f"package {label} SHA-256 {declared} != live {observed}")
        by_id = {receipt.prerequisite_id: receipt for receipt in self.receipts}
        sensitivity_receipt = by_id["input_sensitivity"]
        sensitivity_rows = tuple(
            row for doc_index, document in enumerate(sensitivity_receipt.documents)
            for row in document.rows(label=(
                f"input_sensitivity document {doc_index}/seed={document.suite_seed}")))
        try:
            observations = sensitivity.observations_from_csv_rows(
                sensitivity_rows,
                expected_seeds=tuple(document.suite_seed
                                     for document in sensitivity_receipt.documents))
            sensitivity_report = sensitivity.reduce_input_sensitivity(observations)
        except (TypeError, ValueError) as exc:
            raise SourcePrerequisitePackageError(
                f"input_sensitivity raw evidence is invalid: {exc}") from exc
        if sensitivity_report.suite_version != sensitivity_receipt.suite_version:
            raise SourcePrerequisitePackageError(
                "input_sensitivity reducer suite version differs from its package receipt")

        hostile_receipt = by_id["hostile_distributions"]
        hostile_document = hostile_receipt.documents[0]
        hostile_rows = hostile_document.rows(label="hostile_distributions document")
        hostile_check = oracle_integrity.evaluate_hostile_rows(
            hostile_rows, expected_seed=hostile_document.suite_seed,
            expected_suite_version=hostile_receipt.suite_version)

        checker_receipt = by_id["checker_isolation"]
        checker_document = checker_receipt.documents[0]
        checker_rows = checker_document.rows(label="checker_isolation document")
        checker_check = oracle_integrity.evaluate_checker_rows(
            checker_rows, expected_suite_version=checker_receipt.suite_version)

        def provenance(receipt: PrerequisiteReceipt
                       ) -> source_candidate_authority.EvidenceProvenance:
            return source_candidate_authority.EvidenceProvenance(
                evidence_ref=(f"sha256:{self.package_sha256}#"
                              f"{receipt.prerequisite_id}:{receipt.receipt_sha256}"),
                # The authority-bearing object is the complete package: its
                # header binds the candidate binary/source/evaluator while the
                # fragment identifies the reducer receipt within it. Naming
                # only the receipt here would permit identical CSV bytes to be
                # detached and repackaged under another binary identity.
                evidence_sha256=self.package_sha256,
                evaluator_bundle_sha256=self.evaluator_bundle_sha256,
                capture_mode=self.capture_mode)

        bound = (
            source_candidate_authority.bind_sensitivity(
                sensitivity_report,
                candidate_source_sha256=self.candidate_source_sha256,
                provenance=provenance(sensitivity_receipt)),
            source_candidate_authority.bind_oracle_check(
                "hostile_distributions", hostile_check,
                suite_version=hostile_receipt.suite_version,
                candidate_source_sha256=self.candidate_source_sha256,
                provenance=provenance(hostile_receipt)),
            source_candidate_authority.bind_oracle_check(
                "checker_isolation", checker_check,
                suite_version=checker_receipt.suite_version,
                candidate_source_sha256=self.candidate_source_sha256,
                provenance=provenance(checker_receipt)),
        )
        return tuple(sorted(bound, key=lambda item: item.prerequisite_id))


def _read_package_bytes(path: Any) -> bytes:
    target = os.fspath(Path(path))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise SourcePrerequisitePackageError(
            f"cannot open source prerequisite package {target!r}: {exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise SourcePrerequisitePackageError(
                "source prerequisite package must be one regular, single-link file")
        if info.st_size <= 0 or info.st_size > MAX_PACKAGE_BYTES:
            raise SourcePrerequisitePackageError(
                f"source prerequisite package size must be 1..{MAX_PACKAGE_BYTES} bytes")
        chunks = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise SourcePrerequisitePackageError(
                    "source prerequisite package was truncated during its snapshot read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise SourcePrerequisitePackageError(
                "source prerequisite package grew during its snapshot read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_source_prerequisite_package(path: Any) -> SourcePrerequisitePackage:
    """Snapshot and validate the complete package before any host claim."""
    body = _read_package_bytes(path)
    def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict:
        value = {}
        for key, item in pairs:
            if key in value:
                raise SourcePrerequisitePackageError(
                    f"source prerequisite package repeats JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise SourcePrerequisitePackageError(
            f"source prerequisite package contains non-finite JSON value {value}")

    try:
        payload = json.loads(
            body.decode("utf-8", "strict"),
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePrerequisitePackageError(
            f"source prerequisite package is not strict JSON: {exc}") from exc
    return SourcePrerequisitePackage.from_mapping(payload)


def build_source_prerequisite_package(
        *, campaign_id: str, proposal_id: str, candidate_id: str,
        candidate_source_sha256: str, candidate_binary_sha256: str,
        evaluator_bundle_sha256: str, capture_mode: str,
        documents_by_prerequisite: Mapping[str, tuple[tuple[int, bytes], ...]],
        ) -> SourcePrerequisitePackage:
    """Build the same strict object accepted by the archive/resume loader.

    Fresh producers do not get a second evidence grammar.  Their captured CSV
    bytes enter this constructor and are immediately parsed again through
    ``SourcePrerequisitePackage.from_mapping``; a package created in-process
    therefore has exactly the same identity and reducer boundary as one loaded
    after a restart.
    """
    if set(documents_by_prerequisite) != REQUIRED_IDS:
        raise SourcePrerequisitePackageError(
            f"fresh documents must contain exactly {sorted(REQUIRED_IDS)}")
    receipts = []
    for prerequisite_id in sorted(REQUIRED_IDS):
        documents = []
        for suite_seed, body in documents_by_prerequisite[prerequisite_id]:
            if not isinstance(body, bytes):
                raise SourcePrerequisitePackageError(
                    f"{prerequisite_id} CSV must be bytes, got {type(body).__name__}")
            document = CsvDocument(
                suite_seed=suite_seed,
                csv_sha256=hashlib.sha256(body).hexdigest(),
                csv_bytes=body).to_mapping()
            documents.append(document)
        # Derive the suite version from the raw receipt itself, not from the
        # candidate's branch name or a caller declaration.
        parsed_documents = tuple(CsvDocument.from_mapping(
            item, label=f"fresh.{prerequisite_id}[{index}]")
            for index, item in enumerate(documents))
        if not parsed_documents:
            raise SourcePrerequisitePackageError(
                f"{prerequisite_id} requires non-empty raw CSV documents")
        versions: set[str] = set()
        if prerequisite_id == "input_sensitivity":
            for document in parsed_documents:
                for row in document.rows(label=f"fresh sensitivity seed={document.suite_seed}"):
                    try:
                        versions.add(sensitivity.parse_sensitivity_receipt(
                            row.get("sensitivity_receipt", "")).suite_version)
                    except ValueError as exc:
                        raise SourcePrerequisitePackageError(str(exc)) from exc
        elif prerequisite_id == "hostile_distributions":
            for row in parsed_documents[0].rows(label="fresh hostile distributions"):
                try:
                    versions.add(oracle_integrity.parse_hostile_receipt(
                        row.get("hostile_receipt", "")).suite_version)
                except ValueError as exc:
                    raise SourcePrerequisitePackageError(str(exc)) from exc
        else:
            for row in parsed_documents[0].rows(label="fresh checker isolation"):
                if not (row.get("property_receipt") or row.get("reference_receipt")):
                    continue
                try:
                    versions.add(oracle_integrity.parse_checker_receipt(
                        row.get("checker_receipt", "")).suite_version)
                except ValueError as exc:
                    raise SourcePrerequisitePackageError(str(exc)) from exc
        if len(versions) != 1:
            raise SourcePrerequisitePackageError(
                f"{prerequisite_id} raw CSV has suite versions {sorted(versions)}")
        receipt = {
            "prerequisite_id": prerequisite_id,
            "suite_version": next(iter(versions)),
            "documents": documents,
            "receipt_sha256": "0" * 64,
        }
        receipt["receipt_sha256"] = receipt_sha256(receipt)
        receipts.append(receipt)
    payload = {
        "schema": SCHEMA,
        "campaign_id": campaign_id,
        "proposal_id": proposal_id,
        "candidate_id": candidate_id,
        "candidate_source_sha256": candidate_source_sha256,
        "candidate_binary_sha256": candidate_binary_sha256,
        "evaluator_bundle_sha256": evaluator_bundle_sha256,
        "producer_id": sensitivity.TRUSTED_PRODUCER,
        "capture_mode": capture_mode,
        "receipts": receipts,
        "package_sha256": "0" * 64,
    }
    payload["package_sha256"] = package_sha256(payload)
    return SourcePrerequisitePackage.from_mapping(payload)


def evaluator_source_files() -> tuple[Path, ...]:
    """Files whose exact bytes define this package/reducer authority."""
    return tuple(Path(module.__file__) for module in (
        correctness, oracle_integrity, sensitivity, source_candidate_authority)) + (
            Path(__file__),)


__all__ = [
    "SCHEMA", "MAX_PACKAGE_BYTES", "SourcePrerequisitePackageError",
    "CsvDocument", "PrerequisiteReceipt", "SourcePrerequisitePackage",
    "receipt_sha256", "package_sha256", "load_source_prerequisite_package",
    "build_source_prerequisite_package", "evaluator_source_files",
]
