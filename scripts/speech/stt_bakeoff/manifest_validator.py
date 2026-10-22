"""CS-4.STT1 manifest validator: refuses malformed / unsafe STT corpus manifests.

Deterministic, CPU-only, stdlib-only. Every refusal raises ManifestRefused with a stable
reason code (see REASONS). CLI:

    python3 -m scripts.speech.stt_bakeoff.manifest_validator <manifest.json> [--prior <old.json>]

exits 0 when accepted, 2 on refusal (reason code on stderr).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

LABELS = ("public-anchor", "private")
TRACKS = ("verbatim", "cleaned")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# keys that would carry raw reference text or audio bytes into a manifest (hashes only allowed)
RAW_KEYS = frozenset({
    "text", "reference_text", "reference", "ref_text", "transcript", "raw_text", "ground_truth",
    "audio_bytes", "audio_b64", "audio_base64", "pcm", "pcm_bytes", "samples", "waveform", "wav_bytes",
})
REASONS = (
    "unreadable", "missing_label", "bad_label", "missing_track", "bad_track", "mixed_strata", "mixed_tracks",
    "private_path_in_git_tree", "private_path_missing", "reference_edit_without_bump", "reference_edit_not_logged",
    "raw_content_in_manifest", "bad_pcm_sha256", "bad_reference_sha256", "missing_utterances",
    "duplicate_utterance_id", "confirmation_missing_frozen_hash", "confirmation_frozen_hash_mismatch",
    "missing_rater_label", "bad_rater_count", "missing_reference_version", "missing_field",
    "prior_set_mismatch", "missing_fillers_flag",
)


class ManifestRefused(Exception):
    """Typed refusal with a stable machine-readable reason code."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _in_git_tree(path: str | Path) -> Optional[Path]:
    """Return the directory holding a `.git` (file or dir) at/above `path`, else None."""
    p = Path(path).resolve()
    for d in [p, *p.parents]:
        if (d / ".git").exists():
            return d
    return None


def canonical_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of the manifest excluding `frozen_manifest_sha256` itself."""
    body = {k: v for k, v in manifest.items() if k != "frozen_manifest_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _scan_raw(obj: Any, where: str = "$") -> None:
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if str(k).lower() in RAW_KEYS:
                raise ManifestRefused("raw_content_in_manifest", f"key {k!r} at {where}")
            _scan_raw(v, f"{where}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _scan_raw(v, f"{where}[{i}]")
    elif isinstance(obj, (bytes, bytearray)):
        raise ManifestRefused("raw_content_in_manifest", f"bytes value at {where}")


def validate_manifest(manifest: Mapping[str, Any], prior: Optional[Mapping[str, Any]] = None) -> None:
    """Raise ManifestRefused on the first violation; return None when the manifest is acceptable."""
    if not isinstance(manifest, Mapping):
        raise ManifestRefused("unreadable", "manifest is not an object")
    _scan_raw(manifest)

    label = manifest.get("label")
    if label is None or label == "":
        raise ManifestRefused("missing_label", "every set carries public-anchor or private")
    if label not in LABELS:
        raise ManifestRefused("bad_label", repr(label))
    track = manifest.get("track")
    if track is None:
        raise ManifestRefused("missing_track", "verbatim or cleaned")
    if track not in TRACKS:
        raise ManifestRefused("bad_track", repr(track))
    for f in ("set_id", "language"):
        if not manifest.get(f):
            raise ManifestRefused("missing_field", f)
    rv = manifest.get("reference_version")
    if not isinstance(rv, int) or isinstance(rv, bool) or rv < 1:
        raise ManifestRefused("missing_reference_version", repr(rv))

    utts = manifest.get("utterances")
    if not isinstance(utts, list) or not utts:
        raise ManifestRefused("missing_utterances")
    seen = set()
    for u in utts:
        if not isinstance(u, Mapping) or not u.get("utterance_id"):
            raise ManifestRefused("missing_field", "utterance_id")
        uid = u["utterance_id"]
        if uid in seen:
            raise ManifestRefused("duplicate_utterance_id", str(uid))
        seen.add(uid)
        if not isinstance(u.get("pcm_sha256"), str) or not HEX64.match(u["pcm_sha256"]):
            raise ManifestRefused("bad_pcm_sha256", str(uid))
        if not isinstance(u.get("reference_text_sha256"), str) or not HEX64.match(u["reference_text_sha256"]):
            raise ManifestRefused("bad_reference_sha256", str(uid))
        if not isinstance(u.get("fillers_in_ref"), bool):
            raise ManifestRefused("missing_fillers_flag", f"{uid} (DP-B: fillers_in_ref per utterance)")

    if label == "private":
        paths = [manifest.get("audio_root"), manifest.get("reference_root")]
        paths += [u.get("audio_path") for u in utts] + [u.get("reference_path") for u in utts]
        if not manifest.get("audio_root") or not manifest.get("reference_root"):
            raise ManifestRefused("private_path_missing", "private sets must name audio_root and reference_root")
        for p in paths:
            if p:
                hit = _in_git_tree(p)
                if hit is not None:
                    raise ManifestRefused("private_path_in_git_tree", f"{p} is inside git tree {hit}")

    if manifest.get("confirmation"):
        if not manifest.get("reference_protocol_label"):
            raise ManifestRefused("missing_rater_label", "confirmation set needs reference_protocol_label")
        rc = manifest.get("rater_count")
        if not isinstance(rc, int) or isinstance(rc, bool) or rc < 1:
            raise ManifestRefused("bad_rater_count", repr(rc))
        fz = manifest.get("frozen_manifest_sha256")
        if not fz:
            raise ManifestRefused("confirmation_missing_frozen_hash")
        if fz != canonical_manifest_sha256(manifest):
            raise ManifestRefused("confirmation_frozen_hash_mismatch")

    if prior is not None:
        check_reference_edits(prior, manifest)


def check_reference_edits(prior: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    """A changed reference_text_sha256 requires a strictly higher reference_version and a logged edit pass."""
    if prior.get("set_id") != new.get("set_id"):
        raise ManifestRefused("prior_set_mismatch", f"{prior.get('set_id')!r} vs {new.get('set_id')!r}")
    old = {u["utterance_id"]: u.get("reference_text_sha256") for u in prior.get("utterances", [])}
    changed = [u["utterance_id"] for u in new.get("utterances", [])
               if u["utterance_id"] in old and old[u["utterance_id"]] != u.get("reference_text_sha256")]
    if not changed:
        return
    ov, nv = prior.get("reference_version"), new.get("reference_version")
    if not (isinstance(ov, int) and isinstance(nv, int) and nv > ov):
        raise ManifestRefused("reference_edit_without_bump",
                              f"{len(changed)} reference hash(es) changed (e.g. {changed[0]}) at version {ov}->{nv}")
    log = new.get("edit_log") or []
    if not any(isinstance(e, Mapping) and e.get("reference_version") == nv for e in log):
        raise ManifestRefused("reference_edit_not_logged", f"no edit_log entry for reference_version {nv}")


def check_no_mixed_strata(sets: Iterable[Mapping[str, Any]]) -> None:
    """Refuse any aggregate/pool request over sets that mix labels (public-anchor vs private)
    or tracks (cleaned vs verbatim). Unlabeled sets are refused outright."""
    sets = list(sets)
    labels, tracks = set(), set()
    for s in sets:
        lb = s.get("label")
        if lb not in LABELS:
            raise ManifestRefused("missing_label" if not lb else "bad_label", str(s.get("set_id")))
        labels.add(lb)
        tracks.add(s.get("track"))
    if len(labels) > 1:
        raise ManifestRefused("mixed_strata", "private and public-anchor sets are never pooled")
    if len(tracks) > 1:
        raise ManifestRefused("mixed_tracks", "cleaned and verbatim references are never pooled")


def validate_file(path: str | Path, prior_path: str | Path | None = None) -> dict:
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        pr = json.loads(Path(prior_path).read_text(encoding="utf-8")) if prior_path else None
    except (OSError, ValueError) as e:
        raise ManifestRefused("unreadable", str(e))
    validate_manifest(m, pr)
    return m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="manifest_validator")
    ap.add_argument("manifest")
    ap.add_argument("--prior")
    a = ap.parse_args(argv)
    try:
        validate_file(a.manifest, a.prior)
    except ManifestRefused as e:
        print(f"REFUSED {e.reason}: {e.detail}", file=sys.stderr)
        return 2
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
