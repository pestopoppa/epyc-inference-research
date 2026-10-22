"""scorer.py — one STT scorer for every bake-off arm (CS-8.STT1 (c)-(h)).

Input is a RAW manifest per arm (raw reference, raw hypothesis, chunk indices,
PCM-SHA-256); output is a list of score records shaped for VB-SPEECH-CONV-1B plus an
in-memory per-utterance list. Because the manifest is raw, a normaliser bump is a
rescoring of the same manifest — no re-inference.

Raw manifest (``RAW_MANIFEST_SCHEMA_VERSION``)::

    {
      "schema_version": "stt_bakeoff.raw_manifest/v1",
      "corpus_id": str,                       # corpus name
      "stratum": str,                         # custody stratum: public|selection|confirmation|private…
      "language_mode": "forced" | "auto",
      "segmentation_mode": "none" | "vad_30s",
      "reference_set_id": str (optional),     # defaults to "+".join(sorted set ids)
      "arm": {arm descriptor, see ARM_FIELDS},
      "rows": [{
          "utterance_id": str, "pcm_sha256": 64-hex (decoded PCM, P-STT-1 §1.2),
          "reference_raw": str, "language": str, "set_id": str,
          "reference_track": "standard" | "cleaned" | …,
          "content_stratum": "clean_read" | "noise" | "hesitation_conversational" | …,
          # exactly one of:
          "hypothesis_raw": str,
          "hypothesis_chunks": [str, …], "chunk_indices": [int, …],
          # optional:
          "decode_status": "ok" | "error", "duration_s": float,
          "rewritten_with_reference": bool   # True is REFUSED
      }, …]
    }

Two senses of "stratum" exist and are kept apart: the manifest-level ``stratum`` is the
P-STT-1 §1.4 custody partition (selection / confirmation / private; drives CS-4.STT2
custody), while the row-level ``content_stratum`` is the acoustic condition (clean read,
noise, hesitation/conversational) reported as per-content-stratum WER (e).

No inference, no I/O beyond reading this file for its own hash.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from scripts.kernel_rnd.autokernel import schemas as _ak_schemas
from scripts.kernel_rnd.autokernel.adapters import whisper_stt
from scripts.speech.stt_bakeoff import stt_norm
from scripts.speech.stt_bakeoff.align import DEL, INS, MATCH, SUB, Alignment, align

RAW_MANIFEST_SCHEMA_VERSION = "stt_bakeoff.raw_manifest/v1"
RECORD_SCHEMA_VERSION = "stt_bakeoff.score_record/v1"
RECORD_KIND = "stt_wer"
METRIC = "stt_corpus_wer"
METRIC_DIRECTION = "lower"
UNIT = "percent"

ARM_KIND_OFFLINE = "offline_utterance_final"
ARM_KIND_STREAMING = "streaming_cache_aware"
ARM_KINDS = (ARM_KIND_OFFLINE, ARM_KIND_STREAMING)
LANGUAGE_MODES = ("forced", "auto")
SEGMENTATION_MODES = ("none", "vad_30s")
MAX_UNSEGMENTED_CLIP_S = 30.0
DECODE_STATUSES = ("ok", "error")

CLASS_VOID = "VOID"  # harness fault — not a taxonomy class, never scored as an error
SCOPE_POOLED = "pooled"
SCOPE_SET = "set"
SCOPE_CONTENT_STRATUM = "content_stratum"
WER_KIND = {
    SCOPE_POOLED: "pooled_corpus_wer",
    SCOPE_SET: "per_set_wer",
    SCOPE_CONTENT_STRATUM: "per_content_stratum_wer",
}

#: Arm-identity fields, passed through verbatim from the caller's arm descriptor.
ARM_FIELDS = (
    "arm_id", "arm_kind", "model_family", "runner_id", "runner_commit", "ggml_version",
    "quant", "backend", "device", "residency_proof", "weights_sha256", "decode_options",
    "kv_dtype", "att_context_right", "delay_ms", "lookahead", "chunk_label",
    "algorithmic_delay_ms", "measured_time_to_final_ms", "whisper_parity_gate_ref",
    "unk_ban", "secondary_package_versions",
)
ARM_REQUIRED = ("arm_id", "arm_kind", "runner_id")
STREAMING_REQUIRED = ("chunk_label", "algorithmic_delay_ms")

#: Field list of a VB-SPEECH-CONV-1B STT-arm WER record. Names are a contract with the
#: belief-kernel adapter; do not rename.
VB_SPEECH_CONV_1B_FIELDS = (
    "schema_version", "record_kind", "metric", "metric_direction", "unit",
    "scope", "scope_id", "wer_kind", "wer", "errors", "reference_tokens",
    "substitutions", "deletions", "insertions",
    "n", "n_utterances", "harness_fault_void_count",
    "ci95", "bootstrap", "set_wers", "unweighted_mean_of_set_wers",
    "taxonomy", "taxonomy_check", "correctness_verdict",
    "markers_removed_count", "unknown_marker_count", "numeral_uncovered_count",
    "repetition_envelope", "repetition_check",
    "unk_count", "unk_ban", "unk_ban_violated",
    "fillers_in_ref", "fillers_in_ref_utterances", "de_compound_split_join",
    "column", "candidate", "secondary", "comparable_with",
    "normaliser_id", "normaliser_sha256", "normaliser_contract", "normaliser_properties",
    "scorer_sha256", "manifest_sha256", "corpus_id", "corpus_manifest_sha256",
    "corpus_key", "language", "languages", "language_mode", "segmentation_mode",
    "stratum", "reference_set_id", "reference_track", "confirmation_reuse_n",
) + ARM_FIELDS

#: Per-utterance fields. These NEVER appear in a score record (records carry counts
#: only), which is what lets a confirmation-stratum record be loop-visible.
PER_UTTERANCE_FIELDS = (
    "utterance_id", "pcm_sha256", "set_id", "content_stratum", "language", "class",
    "void_reason", "errors", "reference_tokens", "hypothesis_tokens", "substitutions",
    "deletions", "insertions", "markers_removed_reference", "markers_removed_hypothesis",
    "unknown_markers", "numeral_uncovered_tokens", "unk_count", "fillers_in_ref",
    "de_compound_split_join", "hyp_to_ref_ratio", "reference_normalised",
    "hypothesis_normalised",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ScorerError(ValueError):
    """Malformed input or a refused operation."""


class ReferenceAwareRewriteRefused(ScorerError):
    """A hypothesis that was (or could be) rewritten with knowledge of the reference.

    Prevents: a post-processor that reads the reference and edits the hypothesis toward
    it, which manufactures matches no recogniser produced.
    """


class MixedReferenceTracks(ScorerError):
    """Prevents pooling a standard-reference track with a cleaned-reference track."""


class IncomparableRecords(ScorerError):
    """Prevents pooling/differencing records from different instruments (v1 vs v2
    candidate vs secondary, different normaliser hash, different track or corpus)."""


class ArmKindMismatch(ScorerError):
    """Prevents an offline arm from being scored in a streaming column (or vice versa)."""


class CorpusMismatch(ScorerError):
    """Prevents a paired comparison over different audio (P-STT-1 §1.2 VOIDs it)."""


class WindowVoid(ScorerError):
    """The normaliser failed its in-run contract/property assertions: the window is VOID,
    never a degraded score (P-STT-1 §1.3)."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=None)
def scorer_sha256() -> str:
    """SHA-256 of this file's source bytes."""
    return sha256_hex(Path(__file__).read_bytes())


def manifest_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """SHA-256 of the canonical JSON of the RAW manifest rows (arm-specific)."""
    return sha256_hex(canonical_json(list(rows)))


def reference_text_sha256(reference_raw: str) -> str:
    return sha256_hex(reference_raw.encode("utf-8"))


def corpus_manifest(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """The public corpus manifest: ``(utterance_id, pcm_sha256, reference_text_sha256)``
    sorted by utterance id. Identical across arms on the same corpus, so it is the
    corpus identity used for ``corpus_key`` and for the confirmation reuse counter."""
    out = [{"utterance_id": r["utterance_id"], "pcm_sha256": r["pcm_sha256"],
            "reference_text_sha256": reference_text_sha256(r["reference_raw"])}
           for r in rows]
    return sorted(out, key=lambda d: d["utterance_id"])


def corpus_manifest_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    return sha256_hex(canonical_json(corpus_manifest(rows)))


def corpus_key(corpus_id: str, corpus_manifest_digest: str) -> str:
    """``corpus_id@manifest_sha256[:12]`` (P-STT-1 grammar)."""
    return f"{corpus_id}@{corpus_manifest_digest[:12]}"


# ---------------------------------------------------------------------------
# Bootstrap (Annex S Appendix S-A)
# ---------------------------------------------------------------------------

def _require_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ScorerError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default 'linear' rule), q in [0, 100]."""
    if not sorted_vals:
        raise ScorerError("percentile of an empty sample")
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _validate_counts(errors: Sequence[int], ref_tokens: Sequence[int]) -> None:
    # whisper_stt.pooled_corpus_wer owns the validation contract; calling it validates.
    whisper_stt.pooled_corpus_wer(list(errors), list(ref_tokens))


def bootstrap_pooled_wer(errors: Sequence[int], ref_tokens: Sequence[int], *,
                         resamples: int, seed: int,
                         confidence: float = 0.95) -> Tuple[float, float]:
    """Utterance-level percentile bootstrap of the POOLED WER (percent).

    Each resample draws ``n`` utterances with replacement (``random.Random(seed)``) and
    re-pools numerator and denominator. A normal approximation is FORBIDDEN (§1.4):
    the paired difference distribution is a point mass at zero with a heavy tail.
    """
    _validate_counts(errors, ref_tokens)
    _require_int(resamples, "resamples", minimum=1)
    _require_int(seed, "seed", minimum=0)
    errs, refs = list(errors), list(ref_tokens)
    n = len(errs)
    rng = random.Random(seed)
    stats: List[float] = []
    for _ in range(resamples):
        e = r = 0
        for _ in range(n):
            i = rng.randrange(n)
            e += errs[i]
            r += refs[i]
        stats.append(100.0 * e / r)
    stats.sort()
    tail = (1.0 - confidence) * 50.0
    return (_percentile(stats, tail), _percentile(stats, 100.0 - tail))


def paired_bootstrap_delta(errors_a: Sequence[int], errors_b: Sequence[int],
                           ref_tokens: Sequence[int], *, resamples: int, seed: int,
                           confidence: float = 0.95) -> Dict[str, Any]:
    """Paired ΔWER (B − A, percentage points) over identical utterances.

    Each resample draws utterance indices once and re-pools BOTH arms' numerators
    over the shared denominator (Appendix S-A paired reduction).
    """
    _validate_counts(errors_a, ref_tokens)
    _validate_counts(errors_b, ref_tokens)
    _require_int(resamples, "resamples", minimum=1)
    _require_int(seed, "seed", minimum=0)
    ea, eb, refs = list(errors_a), list(errors_b), list(ref_tokens)
    if not (len(ea) == len(eb) == len(refs)):
        raise CorpusMismatch("paired arrays differ in length")
    n = len(refs)
    point = 100.0 * (sum(eb) - sum(ea)) / sum(refs)
    rng = random.Random(seed)
    stats: List[float] = []
    for _ in range(resamples):
        da = db = r = 0
        for _ in range(n):
            i = rng.randrange(n)
            da += ea[i]
            db += eb[i]
            r += refs[i]
        stats.append(100.0 * (db - da) / r)
    stats.sort()
    tail = (1.0 - confidence) * 50.0
    return {
        "point_delta_pp": point,
        "interval_pp": [_percentile(stats, tail), _percentile(stats, 100.0 - tail)],
        "resamples": resamples, "seed": seed, "confidence": confidence,
        "method": "utterance_paired_percentile_bootstrap",
        "n_paired": n,
        "n_differing_utterances": sum(1 for a, b in zip(ea, eb) if a != b),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_arm(arm: Mapping[str, Any]) -> Dict[str, Any]:
    """Arm descriptor -> full pass-through dict (absent fields are None).

    Refuses: unknown arm_kind; a streaming arm without chunk label / algorithmic delay
    (a streaming WER with no delay is not comparable to anything); a Parakeet-family
    arm with no explicit ``unk_ban`` (the policy must be declared, never defaulted).
    """
    if not isinstance(arm, Mapping):
        raise ScorerError("arm descriptor must be a mapping")
    for k in ARM_REQUIRED:
        if not isinstance(arm.get(k), str) or not arm.get(k):
            raise ScorerError(f"arm descriptor missing {k!r}")
    if arm["arm_kind"] not in ARM_KINDS:
        raise ScorerError(f"arm_kind must be one of {ARM_KINDS}, got {arm['arm_kind']!r}")
    if arm["arm_kind"] == ARM_KIND_STREAMING:
        missing = [k for k in STREAMING_REQUIRED if arm.get(k) is None]
        if missing:
            raise ScorerError(f"streaming arm {arm['arm_id']!r} missing {missing}")
    if is_parakeet_family(arm) and not isinstance(arm.get("unk_ban"), bool):
        raise ScorerError(
            f"Parakeet-family arm {arm['arm_id']!r} must declare unk_ban as a bool")
    for k, v in arm.items():
        if callable(v):
            raise ReferenceAwareRewriteRefused(
                f"arm descriptor field {k!r} is callable; the scorer accepts no hooks")
    return {k: arm.get(k) for k in ARM_FIELDS}


def is_parakeet_family(arm: Mapping[str, Any]) -> bool:
    return str(arm.get("model_family") or "").casefold().startswith("parakeet")


def check_unk_ban_symmetry(arms: Iterable[Mapping[str, Any]]) -> None:
    """(f) The unk-ban policy is applied symmetrically to every Parakeet-family arm.

    Prevents: one Parakeet arm decoding with ``<unk>`` banned and another without, which
    makes their WER gap a decode-policy difference rather than a model difference.
    """
    policies = {}
    for arm in arms:
        if is_parakeet_family(arm):
            ub = arm.get("unk_ban")
            if not isinstance(ub, bool):
                raise ScorerError(f"Parakeet-family arm {arm.get('arm_id')!r} has no unk_ban")
            policies[arm.get("arm_id")] = ub
    if len(set(policies.values())) > 1:
        raise ScorerError(f"unk_ban differs across Parakeet-family arms: {policies}")


def _validate_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row, Mapping):
        raise ScorerError("manifest row must be a mapping")
    for k, v in row.items():
        if callable(v):
            raise ReferenceAwareRewriteRefused(
                f"row field {k!r} is callable; the scorer accepts no rewrite hooks")
    if row.get("rewritten_with_reference"):
        raise ReferenceAwareRewriteRefused(
            f"utterance {row.get('utterance_id')!r} declares rewritten_with_reference; a "
            f"hypothesis edited with knowledge of the reference is not a recognition "
            f"output and is refused")
    for k in ("utterance_id", "reference_raw", "language", "set_id",
              "reference_track", "content_stratum"):
        if not isinstance(row.get(k), str) or (k != "reference_raw" and not row.get(k)):
            raise ScorerError(f"row {row.get('utterance_id')!r}: {k!r} must be a string")
    digest = row.get("pcm_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.match(digest):
        raise ScorerError(f"row {row['utterance_id']!r}: pcm_sha256 not a lowercase sha256")
    if _ak_schemas.is_placeholder_digest(digest):
        raise ScorerError(f"row {row['utterance_id']!r}: pcm_sha256 is a placeholder digest")
    has_raw = "hypothesis_raw" in row
    has_chunks = "hypothesis_chunks" in row
    if has_raw == has_chunks:
        raise ScorerError(f"row {row['utterance_id']!r}: exactly one of hypothesis_raw / "
                          f"hypothesis_chunks is required")
    if row.get("decode_status", "ok") not in DECODE_STATUSES:
        raise ScorerError(f"row {row['utterance_id']!r}: decode_status not in {DECODE_STATUSES}")


def join_chunks(row: Mapping[str, Any], segmentation_mode: str
                ) -> Tuple[Optional[str], Optional[str]]:
    """(d) Join a hypothesis from its chunks by chunk index. Returns (text, void_reason).

    Chunk indices must be exactly ``0..k-1`` (any order in the manifest, joined in index
    order). A gap, duplicate, negative or non-int index, or a chunk/index length
    mismatch is a HARNESS FAULT: the utterance is VOID (counted, excluded) — never a
    recognition error, because a missing chunk is the harness losing audio, not the
    model mishearing it.
    """
    if "hypothesis_raw" in row:
        text = row["hypothesis_raw"]
        if not isinstance(text, str):
            return None, "hypothesis_raw_not_str"
        idx = row.get("chunk_indices")
        if idx is not None and list(idx) != [0]:
            return None, "chunk_indices_without_chunks"
        chunks, indices = [text], [0]
    else:
        chunks = row.get("hypothesis_chunks")
        indices = row.get("chunk_indices")
        if not isinstance(chunks, list) or not isinstance(indices, list):
            return None, "chunks_or_indices_missing"
        if not chunks or len(chunks) != len(indices):
            return None, "chunk_count_mismatch"
        if any(not isinstance(c, str) for c in chunks):
            return None, "chunk_not_str"
        if any(isinstance(i, bool) or not isinstance(i, int) for i in indices):
            return None, "chunk_index_not_int"
        if sorted(indices) != list(range(len(indices))):
            return None, "non_contiguous_chunk_indices"
    k = len(chunks)
    duration = row.get("duration_s")
    if segmentation_mode == "none" and k > 1:
        return None, "chunks_in_unsegmented_mode"
    if (segmentation_mode == "vad_30s" and isinstance(duration, (int, float))
            and not isinstance(duration, bool) and duration > MAX_UNSEGMENTED_CLIP_S
            and k < 2):
        return None, "long_clip_not_segmented"
    ordered = [c for _, c in sorted(zip(indices, chunks), key=lambda p: p[0])]
    return " ".join(ordered), None


# ---------------------------------------------------------------------------
# Per-utterance scoring
# ---------------------------------------------------------------------------

def count_de_compound_split_join(alignment: Alignment) -> int:
    """German compound split/join events in an alignment (v2 candidate class).

    Within each maximal run of non-match columns, a reference token equal to the
    concatenation of 2-3 contiguous hypothesis tokens (split: ``bahnhof`` -> ``bahn
    hof``) or a hypothesis token equal to the concatenation of 2-3 contiguous reference
    tokens (join) counts one event. Reported as its own class BESIDE WER; the
    underlying S/D/I are still counted as errors (nothing is forgiven).
    """
    events = 0
    run: List = []

    def scan(run_pairs) -> int:
        refs = [p.ref_tok for p in run_pairs if p.ref_tok is not None]
        hyps = [p.hyp_tok for p in run_pairs if p.hyp_tok is not None]
        found = 0
        for whole, parts in ((refs, hyps), (hyps, refs)):
            used = set()
            for w in whole:
                hit = False
                for size in (2, 3):
                    for start in range(0, len(parts) - size + 1):
                        span = range(start, start + size)
                        if any(i in used for i in span):
                            continue
                        if "".join(parts[i] for i in span) == w:
                            used.update(span)
                            found += 1
                            hit = True
                            break
                    if hit:
                        break
        return found

    for p in alignment.pairs:
        if p.op == MATCH:
            if run:
                events += scan(run)
                run = []
        else:
            run.append(p)
    if run:
        events += scan(run)
    return events


def score_utterance(row: Mapping[str, Any], *, normaliser_id: str,
                    segmentation_mode: str,
                    repetition_envelope: Optional[float]) -> Dict[str, Any]:
    """Score ONE utterance. Accepts no rewrite hook of any kind; the normaliser is
    called separately on each side with ``(text, lang, side)`` only.

    Class precedence (exactly one, §1.5): VOID (harness fault, outside the taxonomy) >
    decode_error > unknown_marker > numeral_uncovered > empty > repetition_loop > ok.
    """
    _validate_row(row)
    spec = stt_norm.get_normaliser(normaliser_id)
    lang = row["language"]
    out: Dict[str, Any] = {k: None for k in PER_UTTERANCE_FIELDS}
    out.update(utterance_id=row["utterance_id"], pcm_sha256=row["pcm_sha256"],
               set_id=row["set_id"], content_stratum=row["content_stratum"],
               language=lang, errors=0, substitutions=0, deletions=0, insertions=0,
               unknown_markers=[], numeral_uncovered_tokens=[], unk_count=0,
               fillers_in_ref=0, de_compound_split_join=0,
               markers_removed_reference=0, markers_removed_hypothesis=0)

    hyp_text, void_reason = join_chunks(row, segmentation_mode)
    ref = spec.normalise(row["reference_raw"], lang, side=stt_norm.REFERENCE)
    out.update(reference_tokens=len(ref.tokens), reference_normalised=ref.text,
               markers_removed_reference=ref.markers_removed_count,
               fillers_in_ref=stt_norm.count_fillers(ref.tokens, lang))
    if void_reason is None and not ref.tokens:
        void_reason = "empty_reference"
    if void_reason is not None:
        out.update({"class": CLASS_VOID, "void_reason": void_reason})
        return out

    hyp = spec.normalise(hyp_text, lang, side=stt_norm.HYPOTHESIS)
    out.update(hypothesis_tokens=len(hyp.tokens), hypothesis_normalised=hyp.text,
               markers_removed_hypothesis=hyp.markers_removed_count,
               unk_count=hyp.unk_count,
               unknown_markers=list(ref.unknown_markers + hyp.unknown_markers),
               numeral_uncovered_tokens=list(ref.numeral_uncovered + hyp.numeral_uncovered),
               hyp_to_ref_ratio=len(hyp.tokens) / len(ref.tokens))

    al = align(ref.tokens, hyp.tokens)
    out.update(errors=al.errors, substitutions=al.substitutions,
               deletions=al.deletions, insertions=al.insertions)
    if spec.column != "v1" and stt_norm._lang_key(lang) == "de":
        out["de_compound_split_join"] = count_de_compound_split_join(al)

    if row.get("decode_status", "ok") != "ok":
        cls = "decode_error"
    elif out["unknown_markers"]:
        cls = "unknown_marker"
    elif out["numeral_uncovered_tokens"]:
        cls = "numeral_uncovered"
    elif not hyp.tokens:
        cls = "empty"
    elif repetition_envelope is not None and out["hyp_to_ref_ratio"] > repetition_envelope:
        cls = "repetition_loop"
    else:
        cls = "ok"
    out["class"] = cls
    return out


def hyp_to_ref_ratios(per_utterance: Iterable[Mapping[str, Any]]) -> List[float]:
    """Anchor hyp/ref token ratios (ok utterances) — the input to
    ``whisper_stt.derive_repetition_envelope`` for later candidate runs."""
    return [u["hyp_to_ref_ratio"] for u in per_utterance
            if u["class"] == "ok" and u["hyp_to_ref_ratio"]]


# ---------------------------------------------------------------------------
# Arm scoring
# ---------------------------------------------------------------------------

def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise ScorerError("manifest must be a mapping")
    if manifest.get("schema_version") != RAW_MANIFEST_SCHEMA_VERSION:
        raise ScorerError(f"manifest schema_version must be {RAW_MANIFEST_SCHEMA_VERSION!r}")
    for k in ("corpus_id", "stratum"):
        if not isinstance(manifest.get(k), str) or not manifest.get(k):
            raise ScorerError(f"manifest missing {k!r}")
    if manifest.get("language_mode") not in LANGUAGE_MODES:
        raise ScorerError(f"language_mode must be one of {LANGUAGE_MODES}")
    if manifest.get("segmentation_mode") not in SEGMENTATION_MODES:
        raise ScorerError(f"segmentation_mode must be one of {SEGMENTATION_MODES}")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ScorerError("manifest rows must be a non-empty list")
    ids = [r.get("utterance_id") for r in rows if isinstance(r, Mapping)]
    if len(ids) != len(set(ids)):
        raise ScorerError("duplicate utterance_id in manifest")
    tracks = {r.get("reference_track") for r in rows if isinstance(r, Mapping)}
    if len(tracks) > 1:
        raise MixedReferenceTracks(
            f"manifest mixes reference tracks {sorted(map(str, tracks))}; standard and "
            f"cleaned references are scored separately and never pooled")


def _aggregate(utts: Sequence[Mapping[str, Any]], *, resamples: int, seed: int
               ) -> Dict[str, Any]:
    """Counts + pooled WER + bootstrap over a scope (ok utterances only)."""
    taxonomy = {c: 0 for c in whisper_stt.FAILURE_CLASSES}
    void = 0
    for u in utts:
        if u["class"] == CLASS_VOID:
            void += 1
        else:
            taxonomy[u["class"]] += 1
    ok = [u for u in utts if u["class"] == "ok"]
    n_classified = len(utts) - void
    if n_classified > 0:
        tchk = whisper_stt.check_failure_taxonomy(taxonomy, n_utterances=n_classified)
        tax_check = {"outcome": tchk.outcome, "reasons": list(tchk.reasons)}
    else:
        tax_check = {"outcome": "COULD_NOT_CHECK", "reasons": ["no classified utterances"]}
    if ok:
        errs = [u["errors"] for u in ok]
        refs = [u["reference_tokens"] for u in ok]
        wer = whisper_stt.pooled_corpus_wer(errs, refs)
        lo, hi = bootstrap_pooled_wer(errs, refs, resamples=resamples, seed=seed)
        ci = [lo, hi]
    else:
        wer, ci = None, None
    failing = sum(taxonomy[c] for c in whisper_stt.FAILING_CLASSES)
    if failing:
        verdict = "FAIL"
    elif not ok:
        verdict = "COULD_NOT_CHECK"
    else:
        verdict = "PASS"
    return {
        "wer": wer, "ci95": ci,
        "bootstrap": {"method": "utterance_percentile_bootstrap", "resamples": resamples,
                      "seed": seed, "confidence": 0.95, "interval": ci},
        "errors": sum(u["errors"] for u in ok),
        "reference_tokens": sum(u["reference_tokens"] for u in ok),
        "substitutions": sum(u["substitutions"] for u in ok),
        "deletions": sum(u["deletions"] for u in ok),
        "insertions": sum(u["insertions"] for u in ok),
        "n": len(ok), "n_utterances": len(utts), "harness_fault_void_count": void,
        "taxonomy": taxonomy, "taxonomy_check": tax_check, "correctness_verdict": verdict,
        "markers_removed_count": {
            "reference": sum(u["markers_removed_reference"] for u in utts),
            "hypothesis": sum(u["markers_removed_hypothesis"] for u in utts)},
        "unknown_marker_count": taxonomy["unknown_marker"],
        "numeral_uncovered_count": taxonomy["numeral_uncovered"],
        "unk_count": sum(u["unk_count"] for u in utts),
        "fillers_in_ref": sum(u["fillers_in_ref"] for u in utts),
        "fillers_in_ref_utterances": sum(1 for u in utts if u["fillers_in_ref"]),
        "de_compound_split_join_total": sum(u["de_compound_split_join"] or 0 for u in ok),
        "languages": sorted({u["language"] for u in utts}),
    }


def score_arm(manifest: Mapping[str, Any], *, column: str, bootstrap_resamples: int,
              bootstrap_seed: int,
              anchor_hyp_to_ref_ratios: Optional[Sequence[float]] = None
              ) -> Dict[str, Any]:
    """Score one arm's raw manifest under one column (``v1``, ``v2_candidate`` or
    ``secondary``).

    Returns ``{"records": [...], "per_utterance": [...], "summary": {...}}``. The first
    record is the pooled one; then one per set and one per content stratum.

    * The repetition envelope is DERIVED from ``anchor_hyp_to_ref_ratios`` via
      ``whisper_stt.derive_repetition_envelope``; with none supplied the repetition
      check is COULD_NOT_CHECK — it is never defaulted to a literal.
    * The v1 normaliser must pass ``check_normalizer_contract`` and every normaliser
      must pass ``check_normalizer_properties`` over this corpus, else ``WindowVoid``.
    * Bootstrap resample count and seed are required and recorded (no defaults: they
      derive from the campaign seed).
    """
    if column not in stt_norm.COLUMNS:
        raise ScorerError(f"column must be one of {sorted(stt_norm.COLUMNS)}")
    _require_int(bootstrap_resamples, "bootstrap_resamples", minimum=1)
    _require_int(bootstrap_seed, "bootstrap_seed", minimum=0)
    _validate_manifest(manifest)
    arm = validate_arm(manifest.get("arm") or {})
    spec = stt_norm.COLUMNS[column]
    rows = manifest["rows"]
    for r in rows:
        _validate_row(r)

    contract = stt_norm.contract_check(spec.normaliser_id)
    if not spec.candidate and contract.outcome != whisper_stt.schemas.PASS:
        raise WindowVoid(f"normaliser contract failed: {contract.reasons}")
    texts: List[Tuple[str, str]] = []
    for r in rows:
        texts.append((r["reference_raw"], r["language"]))
        if "hypothesis_raw" in r and isinstance(r["hypothesis_raw"], str):
            texts.append((r["hypothesis_raw"], r["language"]))
        for c in r.get("hypothesis_chunks") or []:
            if isinstance(c, str):
                texts.append((c, r["language"]))
    props = stt_norm.check_properties(texts, spec.normaliser_id)
    if props.check.outcome != whisper_stt.schemas.PASS:
        raise WindowVoid(f"normaliser properties failed in-run: {props.check.reasons} "
                         f"{props.violations[:8]}")

    if anchor_hyp_to_ref_ratios is not None:
        envelope = whisper_stt.derive_repetition_envelope(anchor_hyp_to_ref_ratios)
        rep_check = "CHECKED"
    else:
        envelope, rep_check = None, "COULD_NOT_CHECK"

    per_utt = [score_utterance(r, normaliser_id=spec.normaliser_id,
                               segmentation_mode=manifest["segmentation_mode"],
                               repetition_envelope=envelope) for r in rows]

    man_sha = manifest_sha256(rows)
    cm_sha = corpus_manifest_sha256(rows)
    track = rows[0]["reference_track"]
    set_ids = sorted({u["set_id"] for u in per_utt})
    ref_set_id = manifest.get("reference_set_id") or "+".join(set_ids)
    common = {
        "schema_version": RECORD_SCHEMA_VERSION, "record_kind": RECORD_KIND,
        "metric": METRIC, "metric_direction": METRIC_DIRECTION, "unit": UNIT,
        "column": spec.column, "candidate": spec.candidate, "secondary": spec.secondary,
        "comparable_with": [],
        "normaliser_id": spec.normaliser_id,
        "normaliser_sha256": stt_norm.normaliser_sha256(spec.normaliser_id),
        "normaliser_contract": {"outcome": contract.outcome,
                                "reasons": list(contract.reasons)},
        "normaliser_properties": {"outcome": props.check.outcome,
                                  "symmetric": props.symmetric,
                                  "idempotent": props.idempotent,
                                  "deterministic": props.deterministic},
        "scorer_sha256": scorer_sha256(), "manifest_sha256": man_sha,
        "corpus_id": manifest["corpus_id"], "corpus_manifest_sha256": cm_sha,
        "corpus_key": corpus_key(manifest["corpus_id"], cm_sha),
        "language_mode": manifest["language_mode"],
        "segmentation_mode": manifest["segmentation_mode"],
        "stratum": manifest["stratum"], "reference_track": track,
        # 0 on a public/selection stratum (never a confirmation touch); None on a sealed stratum
        # until custody.seal_scoring_run() fills in the real counter, so an unsealed confirmation
        # record cannot pass the belief-kernel writer (it requires an int).
        "confirmation_reuse_n": None if re.split(r"[/:]", str(manifest["stratum"]))[0]
        in ("private", "confirmation") else 0,
        "repetition_envelope": envelope, "repetition_check": rep_check,
        "unk_ban": arm["unk_ban"],
    }
    common.update(arm)

    def make(scope: str, scope_id: str, utts: Sequence[Mapping[str, Any]],
             reference_set_id: str) -> Dict[str, Any]:
        agg = _aggregate(utts, resamples=bootstrap_resamples, seed=bootstrap_seed)
        rec = {k: None for k in VB_SPEECH_CONV_1B_FIELDS}
        rec.update(common)
        rec.update({k: v for k, v in agg.items() if k in rec})
        langs = agg["languages"]
        rec.update(scope=scope, scope_id=scope_id, wer_kind=WER_KIND[scope],
                   reference_set_id=reference_set_id, languages=langs,
                   language=langs[0] if len(langs) == 1 else "mixed",
                   de_compound_split_join=(agg["de_compound_split_join_total"]
                                           if spec.column != "v1" else None),
                   unk_ban_violated=bool(arm["unk_ban"]) and agg["unk_count"] > 0)
        return rec

    set_records = [make(SCOPE_SET, s, [u for u in per_utt if u["set_id"] == s], s)
                   for s in set_ids]
    strata = sorted({u["content_stratum"] for u in per_utt})
    stratum_records = [make(SCOPE_CONTENT_STRATUM, c,
                            [u for u in per_utt if u["content_stratum"] == c], ref_set_id)
                       for c in strata]
    pooled = make(SCOPE_POOLED, "all", per_utt, ref_set_id)
    set_wers = {r["scope_id"]: r["wer"] for r in set_records}
    defined = [w for w in set_wers.values() if w is not None]
    pooled["set_wers"] = set_wers
    pooled["unweighted_mean_of_set_wers"] = (sum(defined) / len(defined)) if defined else None

    return {
        "records": [pooled] + set_records + stratum_records,
        "per_utterance": per_utt,
        "summary": {
            "pooled_corpus_wer": pooled["wer"],
            "unweighted_mean_of_set_wers": pooled["unweighted_mean_of_set_wers"],
            "per_set_wer": set_wers,
            "per_content_stratum_wer": {r["scope_id"]: r["wer"] for r in stratum_records},
            "labels": {
                "pooled_corpus_wer": "P-STT-1 corpus WER: sum(errors)/sum(ref tokens)",
                "unweighted_mean_of_set_wers": "unweighted mean of per-set pooled WERs "
                                               "(NOT corpus WER)",
                "per_set_wer": "pooled WER within each set",
                "per_content_stratum_wer": "pooled WER within each content stratum",
            },
        },
    }


def check_record_fields(record: Mapping[str, Any]) -> List[str]:
    """Missing VB-SPEECH-CONV-1B field names (empty list = conforming shape)."""
    return [f for f in VB_SPEECH_CONV_1B_FIELDS if f not in record]


# ---------------------------------------------------------------------------
# Comparability, pooling, differencing
# ---------------------------------------------------------------------------

def check_comparable(a: Mapping[str, Any], b: Mapping[str, Any], *,
                     same_corpus: bool = False) -> None:
    """Raise ``IncomparableRecords`` unless ``a`` and ``b`` were scored by one instrument.

    Prevents: pooling or differencing a v1 record with a v2-candidate or secondary
    record (a normaliser change is an instrument change, §1.3), two hashes of the same
    normaliser id, a standard with a cleaned reference track, or (``same_corpus``) two
    different corpora.
    """
    ida, idb = a.get("normaliser_id"), b.get("normaliser_id")
    if ida != idb and not (ida in (b.get("comparable_with") or [])
                           and idb in (a.get("comparable_with") or [])):
        raise IncomparableRecords(f"normaliser {ida!r} vs {idb!r}: different instruments")
    if a.get("normaliser_sha256") != b.get("normaliser_sha256"):
        raise IncomparableRecords("normaliser hash differs: different instrument versions")
    if a.get("column") != b.get("column"):
        raise IncomparableRecords(f"column {a.get('column')!r} vs {b.get('column')!r}")
    if a.get("reference_track") != b.get("reference_track"):
        raise MixedReferenceTracks(
            f"reference track {a.get('reference_track')!r} vs {b.get('reference_track')!r}")
    if a.get("scorer_sha256") != b.get("scorer_sha256"):
        raise IncomparableRecords("scorer hash differs")
    if same_corpus and a.get("corpus_key") != b.get("corpus_key"):
        raise IncomparableRecords(f"corpus {a.get('corpus_key')!r} vs {b.get('corpus_key')!r}")


def pool_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pool comparable records: Σ errors / Σ reference tokens (labelled)."""
    if not records:
        raise ScorerError("nothing to pool")
    for r in records[1:]:
        check_comparable(records[0], r)
    errs = [int(r["errors"]) for r in records if r.get("n")]
    refs = [int(r["reference_tokens"]) for r in records if r.get("n")]
    return {"wer_kind": "pooled_corpus_wer",
            "wer": whisper_stt.pooled_corpus_wer(errs, refs),
            "errors": sum(errs), "reference_tokens": sum(refs),
            "normaliser_id": records[0]["normaliser_id"],
            "reference_track": records[0]["reference_track"]}


def difference_records(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """ΔWER b − a in percentage points, only between comparable same-corpus records."""
    check_comparable(a, b, same_corpus=True)
    if a.get("wer") is None or b.get("wer") is None:
        raise ScorerError("cannot difference an undefined WER")
    return b["wer"] - a["wer"]


def paired_bootstrap(result_a: Mapping[str, Any], result_b: Mapping[str, Any], *,
                     resamples: int, seed: int) -> Dict[str, Any]:
    """Paired ΔWER (B − A) between two ``score_arm`` results on the identical corpus.

    Refuses (``CorpusMismatch``) when utterance ids, PCM hashes or normalised reference
    token counts differ — different audio says nothing about the arms (§1.2). Pairs over
    utterances that are ``ok`` in BOTH arms; the dropped count is reported.
    """
    check_comparable(result_a["records"][0], result_b["records"][0], same_corpus=True)
    ua = {u["utterance_id"]: u for u in result_a["per_utterance"]}
    ub = {u["utterance_id"]: u for u in result_b["per_utterance"]}
    if set(ua) != set(ub):
        raise CorpusMismatch("arms cover different utterance ids")
    for k in ua:
        if ua[k]["pcm_sha256"] != ub[k]["pcm_sha256"]:
            raise CorpusMismatch(f"pcm_sha256 differs for {k!r}")
        if ua[k]["reference_tokens"] != ub[k]["reference_tokens"]:
            raise CorpusMismatch(f"reference token count differs for {k!r}")
    ids = sorted(k for k in ua if ua[k]["class"] == "ok" and ub[k]["class"] == "ok")
    if not ids:
        raise ScorerError("no utterance is ok in both arms")
    out = paired_bootstrap_delta([ua[k]["errors"] for k in ids],
                                 [ub[k]["errors"] for k in ids],
                                 [ua[k]["reference_tokens"] for k in ids],
                                 resamples=resamples, seed=seed)
    out["n_unpaired_excluded"] = len(ua) - len(ids)
    out["arm_a"] = result_a["records"][0]["arm_id"]
    out["arm_b"] = result_b["records"][0]["arm_id"]
    return out


# ---------------------------------------------------------------------------
# (g) delay columns, (h) rank tables
# ---------------------------------------------------------------------------

def delay_table(records: Sequence[Mapping[str, Any]], *, arm_kind: str
                ) -> List[Dict[str, Any]]:
    """Rows for one delay column (``offline_utterance_final`` or
    ``streaming_cache_aware``). Raises ``ArmKindMismatch`` on a record of the other
    kind — an offline arm has no streaming delay and is never ranked against one."""
    if arm_kind not in ARM_KINDS:
        raise ScorerError(f"arm_kind must be one of {ARM_KINDS}")
    rows = []
    for r in records:
        if r.get("arm_kind") != arm_kind:
            raise ArmKindMismatch(
                f"arm {r.get('arm_id')!r} is {r.get('arm_kind')!r}; refused in the "
                f"{arm_kind!r} column")
        row = {"arm_id": r["arm_id"], "wer": r.get("wer"), "scope_id": r.get("scope_id"),
               "measured_time_to_final_ms": r.get("measured_time_to_final_ms")}
        if arm_kind == ARM_KIND_STREAMING:
            row.update(chunk_label=r.get("chunk_label"),
                       algorithmic_delay_ms=r.get("algorithmic_delay_ms"),
                       att_context_right=r.get("att_context_right"),
                       delay_ms=r.get("delay_ms"), lookahead=r.get("lookahead"))
        rows.append(row)
    return sorted(rows, key=lambda x: (str(x["arm_id"]), str(x["scope_id"])))


def rank_table(records: Sequence[Mapping[str, Any]], *, set_id: str
               ) -> Dict[str, Any]:
    """Rank arms on one set (lower WER = better; competition ranking, ties share the
    best rank; display order tie-broken by arm id).

    Only records with ``scope == 'set'`` and ``scope_id == set_id`` are used. All must
    be comparable (one instrument). A record whose correctness verdict is FAIL, or whose
    WER is undefined, receives NO rank (§1.5: not a penalised one) and is listed as
    unranked.
    """
    rel = [r for r in records if r.get("scope") == SCOPE_SET and r.get("scope_id") == set_id]
    if not rel:
        raise ScorerError(f"no set-scope records for {set_id!r}")
    for r in rel[1:]:
        check_comparable(rel[0], r)
    arm_ids = [r["arm_id"] for r in rel]
    if len(arm_ids) != len(set(arm_ids)):
        raise ScorerError(f"duplicate arm in rank table for {set_id!r}")
    rankable = sorted((r for r in rel if r.get("wer") is not None
                       and r.get("correctness_verdict") != "FAIL"),
                      key=lambda r: (r["wer"], r["arm_id"]))
    unranked = sorted(r["arm_id"] for r in rel if r not in rankable)
    rows, prev_wer, prev_rank = [], None, 0
    for pos, r in enumerate(rankable, start=1):
        rank = prev_rank if r["wer"] == prev_wer else pos
        rows.append({"rank": rank, "arm_id": r["arm_id"], "wer": r["wer"], "n": r["n"]})
        prev_wer, prev_rank = r["wer"], rank
    return {"set_id": set_id, "normaliser_id": rel[0]["normaliser_id"],
            "column": rel[0]["column"], "stratum": rel[0].get("stratum"),
            "rows": rows, "unranked": unranked}


def rank_delta_public_vs_private(public: Mapping[str, Any], private: Mapping[str, Any]
                                 ) -> Dict[str, Any]:
    """(h) ``private_rank − public_rank`` per arm ranked in both tables (positive =
    the arm does worse on the private set than its public rank suggests). Refuses to
    compare tables built under different normalisers/columns."""
    if public.get("normaliser_id") != private.get("normaliser_id") or \
            public.get("column") != private.get("column"):
        raise IncomparableRecords("rank tables use different normalisers/columns")
    pub = {r["arm_id"]: r["rank"] for r in public["rows"]}
    prv = {r["arm_id"]: r["rank"] for r in private["rows"]}
    both = sorted(set(pub) & set(prv))
    return {
        "label": "rank_delta_public_vs_private = private_rank - public_rank",
        "public_set_id": public.get("set_id"), "private_set_id": private.get("set_id"),
        "deltas": {a: prv[a] - pub[a] for a in both},
        "not_in_both": sorted(set(pub) ^ set(prv)),
    }
