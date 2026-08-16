"""Immutable source-patch artifacts and their guarded worktree application.

The campaign accepts patch *content*, never a branch name or mutable patch path.
The JSON manifest embeds base64 bytes, binds every campaign identity and is
fully loaded and checked before a claim can be acquired.  Mutation is delegated
to :class:`execution.worktree.Worktree`, the only type that may alter source.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from . import schemas
from .evaluator import correctness, integrity
from .execution import chain, instrument_integrity, reward_hack_scan, worktree

__all__ = [
    "SCHEMA_SOURCE_PATCH", "SourceCandidateError", "SourcePatchManifest",
    "AppliedSourceCandidate", "load_source_patch_manifest",
    "apply_source_candidate", "parameter_patch_bundle_sha256",
    "hunk_identities",
]

SCHEMA_SOURCE_PATCH = "epyc.autokernel.source-patch.v1"
SOURCE_TREE = "llama.cpp"
FILE_SCOPE = "<file-scope>"


class SourceCandidateError(RuntimeError):
    """A source artifact cannot safely authorize this campaign mutation."""


_PLAIN_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_:~<>]*$")
_HUNK = re.compile(
    r"^@@ -(?P<old>\d+)(?:,(?P<oldn>\d+))? \+(?P<new>\d+)"
    r"(?:,(?P<newn>\d+))? @@(?P<context>.*)$")
_FUNC = re.compile(r"(?P<name>[A-Za-z_~][A-Za-z0-9_:~<>]*)\s*\([^()]*\)\s*(?:const\s*)?(?:\{|$)")
_TRUNCATED_FUNC = re.compile(
    r"(?P<name>[A-Za-z_~][A-Za-z0-9_:~<>]*)\s*\(\s*$")
_CONTROL_WORDS = frozenset({"if", "for", "while", "switch", "catch"})
_MODE_LINE = re.compile(r"^(?:old|new|deleted file|new file) mode (?P<mode>\d+)$")
_ALLOWED_MODES = frozenset({"100644", "100755"})


def _canonical_scalar(value: Any, label: str) -> Any:
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise SourceCandidateError(f"{label} must be a finite JSON scalar")


def _safe_path(raw: Any, label: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise SourceCandidateError(f"{label} is not a non-empty POSIX path")
    path = PurePosixPath(raw)
    if (path.is_absolute() or raw != path.as_posix()
            or any(part in ("", ".", "..") for part in path.parts)):
        raise SourceCandidateError(f"{label} {raw!r} escapes or is not normalized")
    if raw.startswith("-"):
        raise SourceCandidateError(f"{label} {raw!r} is option-shaped")
    return path.as_posix()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _symbol_from_context(context: str) -> str:
    normalized = context.strip()
    if (_PLAIN_ID.fullmatch(normalized) is not None
            and normalized not in _CONTROL_WORDS):
        return normalized
    matches = list(_FUNC.finditer(normalized))
    if matches:
        return matches[-1].group("name")
    # GNU diff truncates long C/C++ function context at the opening parenthesis.
    # Accept that exact suffix form while continuing to reject control-flow
    # statements as enclosing source symbols.
    match = _TRUNCATED_FUNC.search(normalized)
    if match is not None and match.group("name") not in _CONTROL_WORDS:
        return match.group("name")
    return FILE_SCOPE


def _symbol_from_hunk(context: str, body: Sequence[str]) -> str:
    """Derive a function from source-backed hunk lines before header prose.

    Diff may label a hunk with the preceding function when the hunk begins on
    macros or a template declaration.  Unchanged/deleted body lines, unlike
    caller-authored hunk prose, are checked against the source when applied.
    A single function definition there is therefore the stronger scope signal.
    """
    header_symbol = _symbol_from_context(context)
    body_symbols: list[str] = []
    for line in body:
        # Only leading unchanged source can identify the declaration whose
        # body the hunk enters.  Later context may begin the *next* function.
        if not line or line[0] != " ":
            break
        normalized = line[1:].strip()
        match = _TRUNCATED_FUNC.search(normalized)
        if match is None or match.group("name") in _CONTROL_WORDS:
            continue
        # A body line is stronger than GNU diff's sometimes-stale header only
        # when it is a truncated declaration/definition, not an ordinary call.
        # Require a declaration-like prefix and reject expression punctuation.
        prefix = normalized[:match.start("name")].strip()
        if (not prefix or prefix in _CONTROL_WORDS
                or any(char in prefix for char in "=;{}")):
            continue
        symbol = match.group("name")
        if symbol not in body_symbols:
            body_symbols.append(symbol)
    if len(body_symbols) == 1:
        return body_symbols[0]
    if header_symbol in body_symbols:
        return header_symbol
    if body_symbols:
        return FILE_SCOPE
    return header_symbol


def _hunk_rows(diff_text: str) -> tuple[tuple[str, str, str], ...]:
    """Return stable hunk ids and content-derived enclosing symbols.

    Each id hashes file, old/new ranges, normalized context, and the complete
    hunk body.  Caller labels never enter the identity.
    """
    current_file: Optional[str] = None
    current_header: Optional[str] = None
    current_context = ""
    body: list[str] = []
    rows: list[tuple[str, str, str]] = []

    def flush() -> None:
        nonlocal current_header, current_context, body
        if current_header is None or current_file is None:
            return
        normalized = "\n".join(line.rstrip() for line in body) + "\n"
        material = {
            "file": current_file, "range": current_header,
            "context": " ".join(current_context.split()), "body": normalized,
        }
        rows.append((current_file, f"akhunk:{schemas.content_hash(material)}",
                     _symbol_from_hunk(current_context, body)))
        current_header, current_context, body = None, "", []

    for line in diff_text.splitlines():
        if line.startswith("diff --git a/"):
            flush()
            match = re.match(r"^diff --git a/(.+) b/(.+)$", line)
            if match is None or match.group(1) != match.group(2):
                raise SourceCandidateError("renames/copies are not admissible source patches")
            current_file = _safe_path(match.group(2), "patch path")
            continue
        match = _HUNK.match(line)
        if match:
            flush()
            if current_file is None:
                raise SourceCandidateError("hunk appears before a diff --git header")
            current_header = (
                f"-{match.group('old')},{match.group('oldn') or '1'}"
                f"+{match.group('new')},{match.group('newn') or '1'}")
            current_context = match.group("context").strip()
            continue
        if current_header is not None:
            body.append(line)
    flush()
    if not rows:
        raise SourceCandidateError("source patch contains no accounted hunk")
    return tuple(sorted(rows))


def hunk_identities(diff_text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return stable hunk ids and content-derived enclosing symbols."""
    rows = _hunk_rows(diff_text)
    return (tuple(sorted(row[1] for row in rows)),
            tuple(sorted(set(row[2] for row in rows))))


def _validate_patch_text(
        patch_bytes: bytes,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...],
           Mapping[str, tuple[str, ...]], tuple[str, ...]]:
    try:
        text = patch_bytes.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise SourceCandidateError(f"patch is not strict UTF-8: {exc}") from exc
    if "\x00" in text:
        raise SourceCandidateError("patch contains NUL bytes")
    for line in text.splitlines():
        mode = _MODE_LINE.match(line)
        if mode and mode.group("mode") not in _ALLOWED_MODES:
            raise SourceCandidateError(
                f"patch mode {mode.group('mode')} is not a regular file mode; "
                "symlink/device/FIFO changes are forbidden")
        if line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            raise SourceCandidateError("renames and copies are forbidden in source patches")
    try:
        parsed = integrity.parse_unified_diff(text)
    except Exception as exc:
        raise SourceCandidateError(f"patch is not an accounted unified diff: {exc}") from exc
    if any(item.is_binary for item in parsed.files):
        raise SourceCandidateError("binary patch members are forbidden")
    paths = tuple(sorted(_safe_path(path, "patch path") for path in parsed.paths()))
    rows = _hunk_rows(text)
    hunk_ids = tuple(sorted(row[1] for row in rows))
    symbols = tuple(sorted({row[2] for row in rows}))
    symbols_by_file = {
        path: tuple(sorted({row[2] for row in rows if row[0] == path}))
        for path in paths
    }
    deleted = tuple(sorted(item.path for item in parsed.files if item.is_deleted_file))
    return text, paths, hunk_ids, symbols, symbols_by_file, deleted


@dataclass(frozen=True)
class SourcePatchManifest:
    campaign_id: str
    proposal_id: str
    candidate_id: str
    source_tree: str
    production_base_commit: str
    instrument_commit: str
    change_class: str
    declared_files: tuple[str, ...]
    declared_symbols: Mapping[str, tuple[str, ...]]
    mechanism_id: str
    patch_sha256: str
    patch_bytes: bytes

    def __post_init__(self) -> None:
        if not self.campaign_id.startswith("ak-") or not self.proposal_id.startswith("akp-") \
                or not self.candidate_id.startswith("akc-"):
            raise SourceCandidateError("manifest campaign/proposal/candidate id prefixes are invalid")
        if self.source_tree != SOURCE_TREE:
            raise SourceCandidateError(f"source_tree must be {SOURCE_TREE!r}")
        schemas.require.commit(self.production_base_commit, "production_base_commit",
                               error=SourceCandidateError)
        schemas.require.commit(self.instrument_commit, "instrument_commit",
                               error=SourceCandidateError)
        schemas.require.sha256(self.patch_sha256, "patch_sha256", error=SourceCandidateError)
        if _sha256(self.patch_bytes) != self.patch_sha256:
            raise SourceCandidateError("patch_sha256 does not match the embedded patch bytes")
        if self.change_class not in schemas.CHANGE_CLASSES or self.change_class == "parameter":
            raise SourceCandidateError("source patch change_class must be a non-parameter class")
        if not isinstance(self.declared_files, tuple) or not self.declared_files:
            raise SourceCandidateError("declared_files must be a non-empty tuple")
        files = tuple(sorted({_safe_path(path, "declared_files[]") for path in self.declared_files}))
        object.__setattr__(self, "declared_files", files)
        if set(self.declared_symbols) != set(files):
            raise SourceCandidateError("declared_symbols must have exactly one key per declared file")
        normalized = {}
        for path in files:
            symbols = self.declared_symbols[path]
            if not isinstance(symbols, tuple) or not symbols:
                raise SourceCandidateError(f"declared_symbols[{path!r}] must be non-empty")
            for symbol in symbols:
                if symbol != FILE_SCOPE and (not isinstance(symbol, str) or not _PLAIN_ID.match(symbol)):
                    raise SourceCandidateError(f"declared symbol {symbol!r} is not exact")
            normalized[path] = tuple(sorted(set(symbols)))
        object.__setattr__(self, "declared_symbols", normalized)
        schemas.require.str(self.mechanism_id, "mechanism_id", error=SourceCandidateError)
        text, paths, _hunks, _actual_symbols, actual_by_file, _deleted = \
            _validate_patch_text(self.patch_bytes)
        scan = reward_hack_scan.scan_unified_diff(text)
        prebuild_findings = {
            "phase_detection": scan.phase_detection_findings,
            "capture_replay": scan.capture_replay_findings,
            "content_specialization": scan.content_specialization_findings,
        }
        detected = {name: rows for name, rows in prebuild_findings.items() if rows}
        if detected:
            raise SourceCandidateError(
                "source patch violates the pre-build reward-integrity policy: "
                f"{sorted(detected)}")
        if paths != files:
            raise SourceCandidateError(
                f"patch paths {list(paths)} do not exactly equal declared_files {list(files)}")
        for path in files:
            outside = sorted(set(actual_by_file[path]) - set(normalized[path]))
            if outside:
                raise SourceCandidateError(
                    f"patch hunks in {path!r} derive undeclared enclosing symbol(s) {outside}")
        if not text.endswith("\n"):
            raise SourceCandidateError("patch bytes must end in a newline")

    @property
    def patch_text(self) -> str:
        return self.patch_bytes.decode("utf-8")

    @property
    def patch_bundle_sha256(self) -> str:
        """Content identity of bytes *and* their complete semantic binding."""
        return schemas.content_hash({
            "schema": SCHEMA_SOURCE_PATCH,
            "campaign_id": self.campaign_id,
            "proposal_id": self.proposal_id,
            "candidate_id": self.candidate_id,
            "source_tree": self.source_tree,
            "production_base_commit": self.production_base_commit,
            "instrument_commit": self.instrument_commit,
            "change_class": self.change_class,
            "declared_files": list(self.declared_files),
            "declared_symbols": {
                path: list(self.declared_symbols[path]) for path in self.declared_files
            },
            "mechanism_id": self.mechanism_id,
            "patch_sha256": self.patch_sha256,
            "patch_encoding": "base64",
            "patch_base64": base64.b64encode(self.patch_bytes).decode("ascii"),
        })

    def bind(self, *, proposal: Mapping[str, Any], campaign_id: str,
             candidate_id: str, production_base_commit: str,
             instrument_commit: str) -> None:
        expected = (campaign_id, proposal.get("proposal_id"), candidate_id,
                    production_base_commit, instrument_commit,
                    proposal.get("change_class"))
        got = (self.campaign_id, self.proposal_id, self.candidate_id,
               self.production_base_commit, self.instrument_commit, self.change_class)
        if got != expected:
            raise SourceCandidateError(f"source manifest identity {got!r} != campaign binding {expected!r}")
        change = proposal.get("change")
        declared = change.get("files_and_symbols") if isinstance(change, Mapping) else None
        expected_entries = []
        for path in self.declared_files:
            for symbol in self.declared_symbols[path]:
                expected_entries.append(f"{path}:{symbol}")
        if not isinstance(declared, list) or sorted(set(declared)) != sorted(expected_entries):
            raise SourceCandidateError(
                "proposal.change.files_and_symbols must exactly equal the manifest's "
                f"file:symbol declarations; expected {sorted(expected_entries)!r}")
        if any(path in {p for paths in instrument_integrity.TRANSLATION_UNITS.values()
                        for p in paths} for path in self.declared_files):
            raise SourceCandidateError("a candidate may not patch a reward-instrument translation unit")


def load_source_patch_manifest(path: Any) -> SourcePatchManifest:
    """Load the whole JSON and embedded bytes once, before any host claim."""
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8", "strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceCandidateError(f"source patch manifest is not strict JSON: {exc}") from exc
    required = {
        "schema", "campaign_id", "proposal_id", "candidate_id", "source_tree",
        "production_base_commit", "instrument_commit", "change_class",
        "declared_files", "declared_symbols", "mechanism_id", "patch_sha256",
        "patch_encoding", "patch_base64",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise SourceCandidateError(
            f"source patch manifest fields must be exactly {sorted(required)}")
    if payload["schema"] != SCHEMA_SOURCE_PATCH or payload["patch_encoding"] != "base64":
        raise SourceCandidateError("source patch manifest schema/encoding is unsupported")
    try:
        patch_bytes = base64.b64decode(payload["patch_base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise SourceCandidateError(f"patch_base64 is invalid: {exc}") from exc
    symbols = payload["declared_symbols"]
    if not isinstance(symbols, Mapping):
        raise SourceCandidateError("declared_symbols must be a mapping")
    return SourcePatchManifest(
        campaign_id=payload["campaign_id"], proposal_id=payload["proposal_id"],
        candidate_id=payload["candidate_id"], source_tree=payload["source_tree"],
        production_base_commit=payload["production_base_commit"],
        instrument_commit=payload["instrument_commit"],
        change_class=payload["change_class"],
        declared_files=tuple(payload["declared_files"]),
        declared_symbols={key: tuple(value) for key, value in symbols.items()},
        mechanism_id=payload["mechanism_id"], patch_sha256=payload["patch_sha256"],
        patch_bytes=patch_bytes)


@dataclass(frozen=True)
class AppliedSourceCandidate:
    manifest: SourcePatchManifest
    candidate_commit: str
    diff_text: str
    actual_files: tuple[str, ...]
    actual_hunk_ids: tuple[str, ...]
    actual_symbols: tuple[str, ...]
    commit_argv: tuple[str, ...]
    mutation_receipt: Mapping[str, Any]
    diff_evidence: chain.DiffEvidence


def _assert_regular_paths(root: str, paths: Sequence[str], *,
                          missing_allowed: Sequence[str] = ()) -> None:
    base = os.path.realpath(root)
    allowed = frozenset(missing_allowed)
    for relative in paths:
        current = base
        parts = PurePosixPath(relative).parts
        for index, part in enumerate(parts):
            current = os.path.join(current, part)
            try:
                info = os.lstat(current)
            except FileNotFoundError:
                if relative in allowed:
                    break
                raise SourceCandidateError(f"patched path {relative!r} is missing") from None
            if stat.S_ISLNK(info.st_mode):
                raise SourceCandidateError(f"patched path {relative!r} crosses a symlink")
            if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise SourceCandidateError(f"patched path {relative!r} crosses a non-directory")
            if index == len(parts) - 1:
                if not stat.S_ISREG(info.st_mode):
                    raise SourceCandidateError(f"patched path {relative!r} is not a regular file")
                if info.st_nlink != 1:
                    raise SourceCandidateError(f"patched path {relative!r} has {info.st_nlink} hard links")


def apply_source_candidate(manifest: SourcePatchManifest, *, proposal: Mapping[str, Any],
                           actor: worktree.Worktree) -> AppliedSourceCandidate:
    """Validate, apply, re-derive, and pathspec-commit one source artifact."""
    if not isinstance(manifest, SourcePatchManifest):
        raise TypeError("manifest must be SourcePatchManifest")
    if not isinstance(actor, worktree.Worktree):
        raise TypeError("actor must be an execution.worktree.Worktree")
    if proposal.get("proposal_id") != manifest.proposal_id \
            or proposal.get("change_class") != manifest.change_class:
        raise SourceCandidateError("proposal identity/change_class does not match the source manifest")
    manifest.bind(
        proposal=proposal, campaign_id=manifest.campaign_id,
        candidate_id=manifest.candidate_id,
        production_base_commit=manifest.production_base_commit,
        instrument_commit=manifest.instrument_commit)
    if actor.source_commit != manifest.instrument_commit or actor.head_commit() != actor.source_commit:
        raise SourceCandidateError("actor worktree is not at the manifest's clean instrument base")
    if not actor.is_clean():
        raise SourceCandidateError("actor worktree is dirty before source patch application")
    _text, _paths, _hunks, _symbols, _by_file, deleted = \
        _validate_patch_text(manifest.patch_bytes)
    # Missing paths are valid only for files explicitly created by this patch.
    new_files = tuple(item.path for item in integrity.parse_unified_diff(_text).files
                      if item.is_new_file)
    _assert_regular_paths(actor.path.path, manifest.declared_files,
                          missing_allowed=new_files)
    mutation = actor.apply_patch_bytes(manifest.patch_bytes)
    _assert_regular_paths(actor.path.path, manifest.declared_files,
                          missing_allowed=deleted)
    message = f"{manifest.candidate_id}: {manifest.mechanism_id}"
    commit_argv = actor.commit_argv_for_paths(manifest.declared_files, message)
    commit = actor.commit_paths(manifest.declared_files, message)
    if commit is None:
        raise SourceCandidateError("patch produced no committable source change")
    if not actor.is_clean():
        raise SourceCandidateError("actor worktree is not clean after exact-path commit")
    diff_text = actor.unified_diff_from_source()
    text, paths, hunk_ids, symbols, symbols_by_file, _deleted = \
        _validate_patch_text(diff_text.encode("utf-8"))
    if paths != manifest.declared_files:
        raise SourceCandidateError("committed diff paths differ from the authorized manifest")
    for path in manifest.declared_files:
        outside = sorted(set(symbols_by_file[path]) - set(manifest.declared_symbols[path]))
        if outside:
            raise SourceCandidateError(
                f"committed diff in {path!r} derives undeclared symbols {outside}")
    estimated = int(proposal["change"]["estimated_diff_size"])
    if estimated < 1:
        raise SourceCandidateError("source proposal estimated_diff_size must be positive")
    diff_evidence = chain.diff_policy_evidence(
        diff_text=text, worktree_root=actor.path.path,
        declared_surface_files=manifest.declared_files,
        envelope=correctness.ChangeClassEnvelope(
            change_class=manifest.change_class, max_changed_lines=estimated,
            max_files_touched=len(manifest.declared_files)),
        branch_name=actor.branch.name if actor.branch else "detached",
        commit_argv=commit_argv, record_schema_violations=())
    semantic = correctness.check_semantic_diff_conformance(diff_evidence.policy)
    if semantic.check.outcome != schemas.PASS or diff_evidence.worst.outcome != schemas.PASS:
        reasons = list(semantic.check.reasons) + list(diff_evidence.worst.reasons)
        raise SourceCandidateError("committed source diff violates policy: " + "; ".join(reasons))
    return AppliedSourceCandidate(
        manifest=manifest, candidate_commit=commit, diff_text=text,
        actual_files=paths, actual_hunk_ids=hunk_ids,
        actual_symbols=tuple(sorted(
            f"{path}:{symbol}" for path in symbols_by_file
            for symbol in symbols_by_file[path])),
        commit_argv=commit_argv, mutation_receipt=dict(mutation),
        diff_evidence=diff_evidence)


def parameter_patch_bundle_sha256(*, proposal: Mapping[str, Any], candidate_id: str) -> str:
    """Content identity for the explicit no-source parameter artifact."""
    return schemas.content_hash({
        "schema": "epyc.autokernel.parameter-candidate.v1",
        "candidate_id": candidate_id, "proposal_id": proposal.get("proposal_id"),
        "parameter_surface": proposal.get("change", {}).get("parameter_surface"),
        "source_diff": "empty",
    })
