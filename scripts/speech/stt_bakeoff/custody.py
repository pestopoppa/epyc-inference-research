"""custody.py — confirmation/private-stratum custody for STT scoring (CS-4.STT2).

A scoring run on a stratum labelled ``private`` or ``confirmation`` produces TWO
outputs and nothing else:

1. a **loop-visible aggregate** ``{wer, ci, n, taxonomy, stratum, corpus_key,
   confirmation_reuse_n}`` — counts and intervals only, no per-utterance value;
2. a **sealed per-utterance file** outside every git tree (dir 0700, file 0600),
   readable by the operator and the audit path only, listed in the seal root's
   ``SHA256SUMS``.

Every such scoring increments ``confirmation_reuse_n`` for that corpus (persisted in the
seal root) and appends a line to the seal root's scoring log. Storage follows
OP-PUBLICREPOS: until the operator chooses, the stratum lives under
``/mnt/raid0/llm/data/`` with SHA256SUMS and only ``public_manifest()`` —
``(utterance_id, pcm_sha256, reference_text_sha256)`` — is committed.

Why: a confirmation stratum stops confirming anything once its per-utterance outcomes
reach a planner, AutoKernel or autopilot — the loop can then fit to it (P-STT-1 §1.4:
confirmation utterances MUST NOT appear in planner context). The reuse counter makes
every look at the stratum visible, because each look spends some of its independence.

Tests MUST pass a ``tmp_path`` seal root; the default is the real custody location.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from scripts.speech.stt_bakeoff import scorer

DEFAULT_SEAL_ROOT = "/mnt/raid0/llm/data/stt-cs4-sealed"
SEALED_STRATUM_LABELS = ("private", "confirmation")
FORBIDDEN_CONSUMERS = frozenset({"planner", "autokernel", "autopilot"})
SEALED_READERS = ("operator", "audit")

COUNTERS_FILE = "confirmation_reuse_counters.json"
LOG_FILE = "scoring_log.jsonl"
SUMS_FILE = "SHA256SUMS"
LOCK_FILE = ".custody.lock"

AGGREGATE_FIELDS = ("wer", "ci", "n", "taxonomy", "stratum", "corpus_key",
                    "confirmation_reuse_n")

PathLike = Union[str, os.PathLike]


class CustodyRefused(PermissionError):
    """Base class: a custody rule refused the operation."""


class SealPathInGitTree(CustodyRefused):
    """Prevents sealed per-utterance output landing in a repository (it would be
    committed, pushed to a PUBLIC repo, and read by every loop that reads the tree)."""


class ForbiddenConsumer(CustodyRefused):
    """Prevents per-utterance confirmation output being handed to a planner, AutoKernel
    or autopilot consumer — the consumers that would fit to it."""


class PerUtteranceLeak(CustodyRefused):
    """Prevents a per-utterance value appearing in the loop-visible aggregate."""


class UnsealedConfirmationWrite(CustodyRefused):
    """Prevents per-utterance output for a confirmation/private stratum being written
    anywhere except the sealed path."""


class NotASealedStratum(ValueError):
    """``seal_scoring_run`` called on a stratum that is not confirmation/private."""


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def is_sealed_stratum(stratum: str) -> bool:
    """True for ``private``/``confirmation`` or a label whose first ``/`` or ``:``
    segment is one of them (``confirmation/2026q4``)."""
    if not isinstance(stratum, str):
        return False
    head = re.split(r"[/:]", stratum.strip().casefold(), maxsplit=1)[0]
    return head in SEALED_STRATUM_LABELS


def git_tree_containing(path: PathLike) -> Optional[Path]:
    """The first ancestor-or-self of ``path`` holding a ``.git`` (file or dir), checking
    both the literal absolute path and its symlink-resolved form; None if neither."""
    p = Path(path).expanduser()
    candidates = [Path(os.path.abspath(p)), p.resolve(strict=False)]
    for base in candidates:
        for d in (base, *base.parents):
            if os.path.lexists(d / ".git"):
                return d
    return None


def refuse_git_tree(path: PathLike) -> Path:
    tree = git_tree_containing(path)
    if tree is not None:
        raise SealPathInGitTree(
            f"{path} is inside the git tree at {tree}; sealed per-utterance output must "
            f"live outside every git tree")
    return Path(os.path.abspath(Path(path).expanduser()))


def refuse_consumer(consumer: Optional[str]) -> None:
    if consumer is not None and str(consumer).strip().casefold() in FORBIDDEN_CONSUMERS:
        raise ForbiddenConsumer(
            f"consumer {consumer!r} may not receive per-utterance confirmation output; "
            f"forbidden: {sorted(FORBIDDEN_CONSUMERS)}")


# ---------------------------------------------------------------------------
# Public manifest
# ---------------------------------------------------------------------------

def public_manifest(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """The ONLY committable view of a stratum: ``(utterance_id, pcm_sha256,
    reference_text_sha256)`` per utterance, sorted. No text, no audio."""
    return scorer.corpus_manifest(rows)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _walk(obj: Any) -> Iterable[Tuple[str, Any]]:
    """Yield (key, value) and bare values recursively; key is '' for list members."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield str(k), v
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield "", v
            yield from _walk(v)


def assert_no_per_utterance(aggregate: Mapping[str, Any],
                            per_utterance: Sequence[Mapping[str, Any]]) -> None:
    """Raise ``PerUtteranceLeak`` if any per-utterance field name (anywhere in the
    nesting), utterance id or PCM hash (as a key or a value) appears in ``aggregate``.

    Transcripts are caught by field name, not by value: a short normalised transcript
    (``ok``, ``yes``) would collide with legitimate aggregate keys.
    """
    forbidden_keys = (set(scorer.PER_UTTERANCE_FIELDS) | {"per_utterance"}) - set(
        AGGREGATE_FIELDS)
    forbidden_vals = set()
    for u in per_utterance:
        for k in ("utterance_id", "pcm_sha256"):
            if u.get(k):
                forbidden_vals.add(u[k])
    for key, val in _walk(aggregate):
        if key in forbidden_keys or key in forbidden_vals:
            raise PerUtteranceLeak(f"aggregate carries per-utterance key {key!r}")
        if isinstance(val, str) and val in forbidden_vals:
            raise PerUtteranceLeak("aggregate carries a per-utterance value")


def loop_visible_aggregate(result: Mapping[str, Any], *,
                           confirmation_reuse_n: Optional[int] = None,
                           extra_fields: Sequence[str] = ()) -> Dict[str, Any]:
    """The loop-visible aggregate from a ``scorer.score_arm`` result (pooled record).

    ``extra_fields`` may add pooled-record fields (e.g. ``normaliser_id``). Requesting a
    per-utterance field, or ``per_utterance`` itself, raises ``PerUtteranceLeak``.
    """
    bad = [f for f in extra_fields
           if f in scorer.PER_UTTERANCE_FIELDS or f == "per_utterance"]
    if bad:
        raise PerUtteranceLeak(f"per-utterance fields requested in the aggregate: {bad}")
    pooled = result["records"][0]
    agg = {
        "wer": pooled["wer"], "ci": pooled["ci95"], "n": pooled["n"],
        "taxonomy": dict(pooled["taxonomy"]), "stratum": pooled["stratum"],
        "corpus_key": pooled["corpus_key"],
        "confirmation_reuse_n": confirmation_reuse_n,
    }
    for f in extra_fields:
        if f not in pooled:
            raise KeyError(f"pooled record has no field {f!r}")
        agg[f] = pooled[f]
    assert_no_per_utterance(agg, result.get("per_utterance") or [])
    return agg


# ---------------------------------------------------------------------------
# Seal-root file operations
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private(path: Path, data: bytes, *, exclusive: bool) -> None:
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if exclusive else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _atomic_write_private(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _write_private(tmp, data, exclusive=False)
    os.replace(tmp, path)


def _append_private(path: Path, line: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (line.rstrip("\n") + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._@-]", "_", name)


def file_sha256(path: PathLike) -> str:
    return scorer.sha256_hex(Path(path).read_bytes())


def update_sha256sums(directory: PathLike, entries: Mapping[str, str]) -> Path:
    """Merge ``{relative_path: sha256}`` into ``directory/SHA256SUMS`` (sorted, mode 0600,
    ``sha256sum -c`` format)."""
    d = Path(directory)
    sums = d / SUMS_FILE
    current: Dict[str, str] = {}
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, rel = line.split(None, 1)
                current[rel.strip().lstrip("*")] = digest
    current.update(entries)
    body = "".join(f"{current[k]}  {k}\n" for k in sorted(current))
    _atomic_write_private(sums, body.encode("utf-8"))
    return sums


def write_stratum_sha256sums(directory: PathLike) -> Path:
    """SHA256SUMS over every regular file under a stratum directory (excluding the sums
    file itself). Refuses a directory inside a git tree."""
    d = refuse_git_tree(directory)
    entries = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != SUMS_FILE and not p.name.endswith(".tmp"):
            entries[str(p.relative_to(d))] = file_sha256(p)
    return update_sha256sums(d, entries)


class _Lock:
    def __init__(self, root: Path):
        self.path = root / LOCK_FILE
        self.fd = -1

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def read_reuse_counter(corpus_key: str, *, seal_root: PathLike = DEFAULT_SEAL_ROOT) -> int:
    p = Path(seal_root) / COUNTERS_FILE
    if not p.exists():
        return 0
    return int(json.loads(p.read_text(encoding="utf-8")).get(corpus_key, 0))


# ---------------------------------------------------------------------------
# The two-output split
# ---------------------------------------------------------------------------

def seal_scoring_run(result: Mapping[str, Any], *,
                     seal_root: PathLike = DEFAULT_SEAL_ROOT,
                     consumer: Optional[str] = None,
                     now: Optional[_dt.datetime] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Seal one confirmation/private scoring run.

    Returns ``(aggregate, receipt)``: ``aggregate`` is the loop-visible dict;
    ``receipt`` (sealed path + sha256 + reuse count) is for the operator/audit path and
    must not be forwarded to a loop consumer.

    Refuses: a stratum that is not confirmation/private; a forbidden consumer; a seal
    root inside any git tree.
    """
    pooled = result["records"][0]
    stratum = pooled["stratum"]
    if not is_sealed_stratum(stratum):
        raise NotASealedStratum(f"stratum {stratum!r} is not confirmation/private")
    refuse_consumer(consumer)
    root = refuse_git_tree(seal_root)
    _ensure_dir(root)
    ts = (now or _dt.datetime.now(_dt.timezone.utc)).astimezone(
        _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = pooled["corpus_key"]
    with _Lock(root):
        counters_path = root / COUNTERS_FILE
        counters = (json.loads(counters_path.read_text(encoding="utf-8"))
                    if counters_path.exists() else {})
        reuse_n = int(counters.get(key, 0)) + 1
        counters[key] = reuse_n
        _atomic_write_private(counters_path,
                              json.dumps(counters, sort_keys=True, indent=1).encode("utf-8"))

        corpus_dir = root / _safe(key)
        _ensure_dir(corpus_dir)
        fname = f"{reuse_n:06d}__{_safe(str(pooled['arm_id']))}__{pooled['column']}.jsonl"
        sealed = corpus_dir / fname
        header = {
            "kind": "stt_bakeoff.sealed_per_utterance/v1", "ts": ts,
            "corpus_key": key, "stratum": stratum, "arm_id": pooled["arm_id"],
            "column": pooled["column"], "normaliser_id": pooled["normaliser_id"],
            "normaliser_sha256": pooled["normaliser_sha256"],
            "scorer_sha256": pooled["scorer_sha256"],
            "manifest_sha256": pooled["manifest_sha256"],
            "confirmation_reuse_n": reuse_n,
        }
        lines = [json.dumps(header, sort_keys=True, ensure_ascii=False)]
        lines += [json.dumps(u, sort_keys=True, ensure_ascii=False)
                  for u in result["per_utterance"]]
        _write_private(sealed, ("\n".join(lines) + "\n").encode("utf-8"), exclusive=True)
        digest = file_sha256(sealed)
        update_sha256sums(root, {str(sealed.relative_to(root)): digest})
        _append_private(root / LOG_FILE, json.dumps({
            "ts": ts, "event": "confirmation_scoring", "corpus_key": key,
            "stratum": stratum, "arm_id": pooled["arm_id"], "column": pooled["column"],
            "normaliser_id": pooled["normaliser_id"],
            "normaliser_sha256": pooled["normaliser_sha256"],
            "scorer_sha256": pooled["scorer_sha256"],
            "manifest_sha256": pooled["manifest_sha256"],
            "confirmation_reuse_n": reuse_n, "sealed_file": str(sealed.relative_to(root)),
            "sealed_sha256": digest, "consumer": consumer,
        }, sort_keys=True))
    aggregate = loop_visible_aggregate(result, confirmation_reuse_n=reuse_n)
    receipt = {"sealed_path": str(sealed), "sealed_sha256": digest,
               "confirmation_reuse_n": reuse_n, "seal_root": str(root)}
    return aggregate, receipt


def write_per_utterance(result: Mapping[str, Any], path: PathLike, *,
                        consumer: Optional[str] = None) -> Path:
    """Write per-utterance output for a NON-sealed stratum (public / selection).

    Refuses (``UnsealedConfirmationWrite``) when the stratum is confirmation/private:
    those go only through ``seal_scoring_run``.
    """
    stratum = result["records"][0]["stratum"]
    if is_sealed_stratum(stratum):
        refuse_consumer(consumer)
        raise UnsealedConfirmationWrite(
            f"stratum {stratum!r} is sealed; per-utterance output goes only through "
            f"seal_scoring_run into the seal root")
    p = Path(path)
    p.write_text("".join(json.dumps(u, sort_keys=True, ensure_ascii=False) + "\n"
                         for u in result["per_utterance"]), encoding="utf-8")
    return p


def read_sealed(path: PathLike, *, reader: str,
                seal_root: PathLike = DEFAULT_SEAL_ROOT) -> List[Dict[str, Any]]:
    """Read a sealed file — operator or audit path only — verifying it against the
    seal root's SHA256SUMS (a modified sealed file is refused, not read)."""
    if reader not in SEALED_READERS:
        refuse_consumer(reader)
        raise CustodyRefused(f"reader {reader!r} not in {SEALED_READERS}")
    root = Path(seal_root)
    p = Path(path)
    rel = str(p.resolve().relative_to(root.resolve()))
    sums = {}
    for line in (root / SUMS_FILE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d, r = line.split(None, 1)
            sums[r.strip()] = d
    if sums.get(rel) != file_sha256(p):
        raise CustodyRefused(f"{rel} does not match SHA256SUMS")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
