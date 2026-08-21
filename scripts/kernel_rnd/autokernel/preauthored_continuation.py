"""Strict immutable authority for a reviewed preauthored source continuation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping

from . import source_candidate


SCHEMA = "epyc.autokernel.preauthored_source_continuation.v1"
DEFAULT_CARRIER = Path(__file__).with_name("preauthored_q5_continuation_v1.json")
# Set only after the reviewed carrier is generated.  The self-hash proves
# internal consistency; this product constant makes coherent substitution
# impossible.
EXPECTED_CARRIER_SHA256 = "819d16c0903d71649c4674080d2718159d12ea1769e1f4f943d04dc7e2974889"
_EXPECTED = {
    "hypothesis_id": "akh-v2-q5-onewave-preauthored",
    "source_file": "ggml/src/ggml-cuda/mmvq.cu",
    "historical_commit": "eb26918fa82f8aef3ab72f1e3263bd8fecde62e7",
    "historical_parent_commit": "e1cbca9fcbc0ed81164c5532b94cd106a83d7368",
    "historical_tree": "a723f77d3666987318f017228a993d610f2b44b1",
    "historical_parent_tree": "1a3c38a26f4d569b48679d16633078bd36900be5",
    "patch_sha256": "f4cc49cd11cdfd93a2d5d2e00e653f503b6a16ce675bfb12c034fbbfae3e7a77",
    "source_backed_diff_sha256": "2adf93c7af423debf39307a3e4d6fa675d5061c565f36682d0b22295df4339c9",
    "mechanism_id": "q5_0_one_wave_per_output_block",
    "template_id": "cuda-mmvq-q5-onewave-continuation-v1",
    "instrument_commit": "5bbcc5498e4732162356953b7be96a53073a6706",
    "preimage_blob": "9f5927771d3cbd21ffbb007df22184868ce7ffa4",
    "preimage_sha256": "15d25d71c945de19e8efc9fbfc6b7e5e66f33bc7635f9dc648d9e1f231ba409e",
    "candidate_blob": "8a605f0088a39f93ac87b23239b8e59310f6de99",
    "candidate_sha256": "3d61c415af15507f6e78e48809cda98ba20b7ef6873b6a2196e9640fa4d0ee39",
    "historical_binary_sha256": "e6540dc80ae41f28cd2791e13e65d12aa9ebba83f63aeba8b48adcc540aec378",
}
_GIT_ENV_REDIRECTS = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES", "GIT_PREFIX", "GIT_INDEX_VERSION",
})
HASH = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
GIT_OBJECT = re.compile(r"^[0-9a-f]{40}$")


class PreauthoredContinuationError(ValueError):
    """The continuation carrier is malformed or no longer byte-authoritative."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreauthoredContinuationError(
            "continuation carrier is not canonicalizable") from exc


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise PreauthoredContinuationError(f"{label} has unknown or missing keys")
    return value


def _text(value: object, label: str) -> str:
    if (not isinstance(value, str) or not value or "\x00" in value
            or value != value.strip()):
        raise PreauthoredContinuationError(f"{label} is not exact non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or HASH.fullmatch(value) is None:
        raise PreauthoredContinuationError(f"{label} is not a SHA-256")
    return value


def _commit(value: object, label: str) -> str:
    if not isinstance(value, str) or COMMIT.fullmatch(value) is None:
        raise PreauthoredContinuationError(f"{label} is not a full Git commit")
    return value


def _git_object(value: object, label: str) -> str:
    if not isinstance(value, str) or GIT_OBJECT.fullmatch(value) is None:
        raise PreauthoredContinuationError(f"{label} is not a full Git object")
    return value


def _path(value: object, label: str) -> str:
    text = _text(value, label)
    parsed = PurePosixPath(text)
    if (parsed.is_absolute() or parsed.as_posix() != text
            or any(part in {"", ".", ".."} for part in parsed.parts)):
        raise PreauthoredContinuationError(f"{label} is not a normalized relative path")
    return text


@dataclass(frozen=True)
class PreauthoredContinuation:
    hypothesis_id: str
    source_tree: str
    source_file: str
    historical_commit: str
    historical_parent_commit: str
    historical_tree: str
    historical_parent_tree: str
    patch_sha256: str
    patch_bytes: bytes
    source_backed_diff_sha256: str
    source_backed_diff: str
    declared_symbols: tuple[str, ...]
    mechanism_id: str
    change_class: str
    compatibility_bridge: Mapping[str, Any]
    template_id: str
    correctness_id: str
    dispatch_id: str
    expected_dispatch: tuple[Mapping[str, Any], ...]
    excluded_dispatch: tuple[Mapping[str, Any], ...]
    historical_receipts: tuple[Mapping[str, Any], ...]
    correctness_policy: Mapping[str, Any]
    sha256: str


def validate(body: object) -> PreauthoredContinuation:
    top = _exact(body, {
        "schema", "hypothesis_id", "source", "historical_candidate", "patch",
        "compatibility_bridge", "experiment_intent", "historical_receipts",
        "correctness_policy", "carrier_sha256",
    }, "continuation carrier")
    if top["schema"] != SCHEMA:
        raise PreauthoredContinuationError("continuation carrier schema mismatch")
    carrier_sha = _digest(top["carrier_sha256"], "carrier_sha256")
    expected_carrier_sha = hashlib.sha256(_canonical({
        key: value for key, value in top.items() if key != "carrier_sha256"
    })).hexdigest()
    if carrier_sha != expected_carrier_sha:
        raise PreauthoredContinuationError("continuation carrier self-hash mismatch")
    hypothesis_id = _text(top["hypothesis_id"], "hypothesis_id")
    if hypothesis_id != _EXPECTED["hypothesis_id"]:
        raise PreauthoredContinuationError("continuation hypothesis id is invalid")

    source = _exact(top["source"], {"tree", "file", "declared_symbols",
                                     "mechanism_id", "change_class"}, "source")
    source_tree = _text(source["tree"], "source.tree")
    if source_tree != "llama.cpp":
        raise PreauthoredContinuationError("continuation source tree is not llama.cpp")
    source_file = _path(source["file"], "source.file")
    if source_file != _EXPECTED["source_file"]:
        raise PreauthoredContinuationError("continuation source file changed")
    symbols = source["declared_symbols"]
    if (not isinstance(symbols, list) or len(symbols) != 4
            or symbols != sorted(set(symbols))
            or any(not isinstance(item, str) or not item for item in symbols)):
        raise PreauthoredContinuationError(
            "continuation requires four sorted distinct declared symbols")
    mechanism_id = _text(source["mechanism_id"], "source.mechanism_id")
    if mechanism_id != _EXPECTED["mechanism_id"]:
        raise PreauthoredContinuationError(
            "continuation mechanism identity changed")
    change_class = _text(source["change_class"], "source.change_class")

    historical = _exact(top["historical_candidate"], {
        "commit", "parent_commit", "tree", "parent_tree",
        "candidate_file_git_blob", "candidate_file_sha256",
    }, "historical_candidate")
    historical_commit = _commit(historical["commit"], "historical_candidate.commit")
    historical_parent_commit = _commit(
        historical["parent_commit"], "historical_candidate.parent_commit")
    historical_tree = _git_object(historical["tree"], "historical_candidate.tree")
    historical_parent_tree = _git_object(
        historical["parent_tree"], "historical_candidate.parent_tree")
    _git_object(historical["candidate_file_git_blob"],
                "historical_candidate.candidate_file_git_blob")
    _digest(historical["candidate_file_sha256"],
            "historical_candidate.candidate_file_sha256")
    if (historical_commit != _EXPECTED["historical_commit"]
            or historical_parent_commit != _EXPECTED["historical_parent_commit"]
            or historical_tree != _EXPECTED["historical_tree"]
            or historical_parent_tree != _EXPECTED["historical_parent_tree"]
            or historical["candidate_file_git_blob"] != _EXPECTED["candidate_blob"]
            or historical["candidate_file_sha256"] != _EXPECTED["candidate_sha256"]):
        raise PreauthoredContinuationError(
            "historical candidate identity differs from reviewed authority")

    patch = _exact(top["patch"], {
        "sha256", "base64", "source_backed_sha256",
        "source_backed_base64",
    }, "patch")
    patch_sha = _digest(patch["sha256"], "patch.sha256")
    try:
        patch_bytes = base64.b64decode(patch["base64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise PreauthoredContinuationError("patch.base64 is invalid") from exc
    if not patch_bytes or hashlib.sha256(patch_bytes).hexdigest() != patch_sha:
        raise PreauthoredContinuationError("embedded patch bytes changed")
    if patch_sha != _EXPECTED["patch_sha256"]:
        raise PreauthoredContinuationError("continuation patch identity changed")
    source_backed_sha = _digest(
        patch["source_backed_sha256"], "patch.source_backed_sha256")
    try:
        source_backed_bytes = base64.b64decode(
            patch["source_backed_base64"], validate=True)
        source_backed_diff = source_backed_bytes.decode("utf-8", "strict")
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise PreauthoredContinuationError(
            "patch.source_backed_base64 is invalid") from exc
    if (not source_backed_diff.endswith("\n")
            or hashlib.sha256(source_backed_bytes).hexdigest()
               != source_backed_sha
            or source_backed_sha != _EXPECTED["source_backed_diff_sha256"]
            or source_candidate.source_backed_symbol_map(source_backed_diff)
               != {source_file: tuple(symbols)}):
        raise PreauthoredContinuationError(
            "source-backed diff authority changed")

    bridge = _exact(top["compatibility_bridge"], {
        "policy", "historical_parent_file_git_blob",
        "historical_parent_file_sha256", "current_instrument_commit",
        "current_instrument_file_git_blob", "current_instrument_file_sha256",
        "patched_file_git_blob", "patched_file_sha256",
    }, "compatibility_bridge")
    if bridge["policy"] != "exact_preimage_bytes_and_patch_output_v1":
        raise PreauthoredContinuationError("compatibility bridge policy mismatch")
    _commit(bridge["current_instrument_commit"],
            "compatibility_bridge.current_instrument_commit")
    for key in ("historical_parent_file_git_blob", "current_instrument_file_git_blob",
                "patched_file_git_blob"):
        _git_object(bridge[key], f"compatibility_bridge.{key}")
    for key in ("historical_parent_file_sha256", "current_instrument_file_sha256",
                "patched_file_sha256"):
        _digest(bridge[key], f"compatibility_bridge.{key}")
    if (bridge["historical_parent_file_git_blob"]
            != bridge["current_instrument_file_git_blob"]
            or bridge["historical_parent_file_sha256"]
            != bridge["current_instrument_file_sha256"]
            or bridge["patched_file_git_blob"]
            != historical["candidate_file_git_blob"]
            or bridge["patched_file_sha256"]
            != historical["candidate_file_sha256"]):
        raise PreauthoredContinuationError(
            "compatibility bridge does not prove the same preimage and output")
    if (bridge["current_instrument_commit"] != _EXPECTED["instrument_commit"]
            or bridge["historical_parent_file_git_blob"] != _EXPECTED["preimage_blob"]
            or bridge["historical_parent_file_sha256"] != _EXPECTED["preimage_sha256"]
            or bridge["patched_file_git_blob"] != _EXPECTED["candidate_blob"]
            or bridge["patched_file_sha256"] != _EXPECTED["candidate_sha256"]):
        raise PreauthoredContinuationError("compatibility bridge authority changed")

    intent = _exact(top["experiment_intent"], {
        "template_id", "correctness_id", "dispatch_id", "expected_dispatch",
        "excluded_dispatch",
    }, "experiment_intent")
    template_id = _text(intent["template_id"], "experiment_intent.template_id")
    correctness_id = _text(intent["correctness_id"], "experiment_intent.correctness_id")
    dispatch_id = _text(intent["dispatch_id"], "experiment_intent.dispatch_id")
    if (template_id != _EXPECTED["template_id"]
            or correctness_id != "backend-ops-hip-v1"
            or dispatch_id != "decode-tg128-rocprof-v3"):
        raise PreauthoredContinuationError("continuation experiment identity changed")
    route_keys = {"route_id", "kernel_name", "calls", "grid", "workgroup", "lds_bytes"}
    routes: list[tuple[Mapping[str, Any], ...]] = []
    for field, expected_count in (("expected_dispatch", 3), ("excluded_dispatch", 1)):
        rows = intent[field]
        if (not isinstance(rows, list) or len(rows) != expected_count
                or any(not isinstance(row, Mapping) or set(row) != route_keys
                       for row in rows)):
            raise PreauthoredContinuationError(f"{field} route authority is malformed")
        checked: list[Mapping[str, Any]] = []
        for row in rows:
            route = dict(row)
            for key in ("route_id", "kernel_name"):
                _text(route[key], f"{field}.{key}")
            for key in ("calls", "grid", "workgroup", "lds_bytes"):
                value = route[key]
                if (isinstance(value, bool) or not isinstance(value, int)
                        or value < 0 or key != "lds_bytes" and value == 0):
                    raise PreauthoredContinuationError(
                        f"{field}.{key} is outside bounded integer authority")
            checked.append(route)
        routes.append(tuple(checked))

    receipts = top["historical_receipts"]
    receipt_keys = {
        "receipt_evidence_id", "stdout_evidence_id", "stderr_evidence_id",
        "binary_evidence_id", "file_sha256", "schema", "result",
        "source_commit", "binary_sha256", "stdout_sha256", "stderr_sha256",
        "scope",
    }
    if (not isinstance(receipts, list) or len(receipts) != 2
            or any(not isinstance(row, Mapping) or set(row) != receipt_keys
                   for row in receipts)):
        raise PreauthoredContinuationError("historical receipt authority is malformed")
    checked_receipts: list[Mapping[str, Any]] = []
    for row in receipts:
        receipt = dict(row)
        for key in ("file_sha256", "binary_sha256", "stdout_sha256", "stderr_sha256"):
            _digest(receipt[key], f"historical_receipts.{key}")
        if (receipt["schema"] != "epyc.autokernel.targeted_correctness_receipt.v1"
                or receipt["result"] != "PASS"
                or receipt["source_commit"] != historical_commit[:9]
                or receipt["binary_sha256"]
                   != _EXPECTED["historical_binary_sha256"]
                or receipt["scope"] not in {"targeted_q5_0", "full_backend_ops"}):
            raise PreauthoredContinuationError(
                "historical receipt semantic identity is not exact")
        checked_receipts.append(receipt)
    if ({row["scope"] for row in checked_receipts}
            != {"targeted_q5_0", "full_backend_ops"}
            or len({row["receipt_evidence_id"] for row in checked_receipts}) != 2
            or {row["binary_evidence_id"] for row in checked_receipts}
               != {"ev-q5-onewave-correctness-binary"}):
        raise PreauthoredContinuationError(
            "historical receipt coverage is incomplete")

    correctness = _exact(top["correctness_policy"], {
        "historical_receipts_authority", "modern_governed_correctness",
        "bridge_waives_current_correctness", "scientific_boundary",
    }, "correctness_policy")
    if correctness != {
        "historical_receipts_authority": "provenance_only",
        "modern_governed_correctness": "required_after_current_instrument_build",
        "bridge_waives_current_correctness": False,
        "scientific_boundary": "dispatch_attribution",
    }:
        raise PreauthoredContinuationError("correctness policy was weakened")

    return PreauthoredContinuation(
        hypothesis_id=hypothesis_id, source_tree=source_tree,
        source_file=source_file, historical_commit=historical_commit,
        historical_parent_commit=historical_parent_commit,
        historical_tree=historical_tree,
        historical_parent_tree=historical_parent_tree,
        patch_sha256=patch_sha, patch_bytes=patch_bytes,
        source_backed_diff_sha256=source_backed_sha,
        source_backed_diff=source_backed_diff,
        declared_symbols=tuple(symbols), mechanism_id=mechanism_id,
        change_class=change_class, compatibility_bridge=dict(bridge),
        template_id=template_id, correctness_id=correctness_id,
        dispatch_id=dispatch_id, expected_dispatch=routes[0],
        excluded_dispatch=routes[1],
        historical_receipts=tuple(checked_receipts),
        correctness_policy=dict(correctness), sha256=carrier_sha)


def verify_git_authority(value: PreauthoredContinuation, repository: Path,
                         instrument_commit: str) -> str:
    """Reconstruct the historical patch and current byte bridge from Git."""
    if (not isinstance(value, PreauthoredContinuation)
            or instrument_commit != value.compatibility_bridge[
                "current_instrument_commit"]):
        raise PreauthoredContinuationError(
            "continuation instrument authority changed")
    repository = repository.resolve(strict=True)
    git_env = {
        key: value for key, value in os.environ.items()
        if key not in _GIT_ENV_REDIRECTS and not key.startswith("GIT_CONFIG")
    }
    git_env.update({
        "PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C",
        "GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
    })

    def git(*args: str) -> bytes:
        result = subprocess.run(
            ("/usr/bin/git", "-C", str(repository),
             "-c", "diff.external=", "-c", "core.attributesfile=/dev/null",
             *args),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, env=git_env)
        if result.returncode:
            raise PreauthoredContinuationError(
                "continuation Git authority cannot be reconstructed")
        return result.stdout

    raw_commit = git("cat-file", "commit", value.historical_commit).decode(
        "utf-8", "strict")
    header = raw_commit.split("\n\n", 1)[0].splitlines()
    trees = [line.removeprefix("tree ") for line in header
             if line.startswith("tree ")]
    parents = [line.removeprefix("parent ") for line in header
               if line.startswith("parent ")]
    if trees != [value.historical_tree] \
            or parents != [value.historical_parent_commit]:
        raise PreauthoredContinuationError(
            "historical candidate does not have the exact single parent")
    if (git("rev-parse", f"{value.historical_commit}^{{tree}}").strip().decode()
            != value.historical_tree
            or git("rev-parse", f"{value.historical_parent_commit}^{{tree}}").strip().decode()
            != value.historical_parent_tree):
        raise PreauthoredContinuationError("historical tree identity changed")
    paths = git("diff", "--name-only", "--no-ext-diff", "--no-textconv",
                value.historical_parent_commit, value.historical_commit).decode(
                    "utf-8", "strict").splitlines()
    if paths != [value.source_file]:
        raise PreauthoredContinuationError(
            "historical candidate changed outside its one reviewed file")
    patch = git("diff", "--binary", "--no-ext-diff", "--no-textconv",
                value.historical_parent_commit, value.historical_commit, "--",
                value.source_file)
    if patch != value.patch_bytes:
        raise PreauthoredContinuationError(
            "historical Git diff differs from embedded patch bytes")
    source_backed_diff = git(
        "diff", "--no-ext-diff", "--no-textconv", "--unified=3",
        "--function-context", value.historical_parent_commit,
        value.historical_commit, "--", value.source_file).decode(
            "utf-8", "strict")
    if (source_backed_diff != value.source_backed_diff
            or hashlib.sha256(source_backed_diff.encode("utf-8")).hexdigest()
               != value.source_backed_diff_sha256
            or source_candidate.source_backed_symbol_map(source_backed_diff) != {
                value.source_file: value.declared_symbols}):
        raise PreauthoredContinuationError(
            "historical function-context diff changed source-backed symbols")
    bridge = value.compatibility_bridge
    for commit, blob_key, digest_key in (
            (value.historical_parent_commit,
             "historical_parent_file_git_blob", "historical_parent_file_sha256"),
            (instrument_commit,
             "current_instrument_file_git_blob", "current_instrument_file_sha256"),
            (value.historical_commit,
             "patched_file_git_blob", "patched_file_sha256")):
        blob = git("rev-parse", f"{commit}:{value.source_file}").strip().decode()
        content = git("show", f"{commit}:{value.source_file}")
        if (blob != bridge[blob_key]
                or hashlib.sha256(content).hexdigest() != bridge[digest_key]):
            raise PreauthoredContinuationError(
                "continuation file/blob bridge no longer reconstructs")
    return source_backed_diff


def load(path: os.PathLike[str] | str = DEFAULT_CARRIER) -> PreauthoredContinuation:
    source = Path(path)
    try:
        descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_mode & 0o022):
            raise PreauthoredContinuationError(
                "continuation carrier lacks single-link read-only authority")
        with os.fdopen(descriptor, "rb") as handle:
            raw = handle.read()
            after = os.fstat(handle.fileno())
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
             before.st_ctime_ns, before.st_nlink)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                    after.st_ctime_ns, after.st_nlink)):
            raise PreauthoredContinuationError(
                "continuation carrier changed while read")
        body = json.loads(
            raw.decode("utf-8", "strict"), object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PreauthoredContinuationError(
                    f"continuation carrier contains non-finite {value}")))
        if raw != _canonical(body) + b"\n":
            raise PreauthoredContinuationError(
                "continuation carrier bytes are not canonical JSON")
        current = source.stat(follow_symlinks=False)
        if ((before.st_dev, before.st_ino, before.st_uid, before.st_mode,
             before.st_size, before.st_mtime_ns, before.st_ctime_ns,
             before.st_nlink)
                != (current.st_dev, current.st_ino, current.st_uid,
                    current.st_mode, current.st_size, current.st_mtime_ns,
                    current.st_ctime_ns, current.st_nlink)
                or before.st_uid != os.geteuid()):
            raise PreauthoredContinuationError(
                "continuation carrier pathname identity changed")
    except PreauthoredContinuationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreauthoredContinuationError(
            "continuation carrier is unreadable JSON") from exc
    result = validate(body)
    if result.sha256 != EXPECTED_CARRIER_SHA256:
        raise PreauthoredContinuationError(
            "continuation carrier is not the reviewed product authority")
    return result


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreauthoredContinuationError(
                f"continuation carrier contains duplicate key: {key}")
        result[key] = value
    return result


__all__ = [
    "SCHEMA", "DEFAULT_CARRIER", "EXPECTED_CARRIER_SHA256",
    "PreauthoredContinuationError",
    "PreauthoredContinuation", "validate", "load", "verify_git_authority",
]
