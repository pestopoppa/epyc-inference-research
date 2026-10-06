"""KB-RAG query-length instrumentation (internal-kb-rag.md H2).

Before this module, no instrument anywhere in the retrieval path could say whether a
live agent query had EVER exceeded the 48-token query cap. Every statement about the
caps was inferred from a curated 90-case pool. This module is the write side and the
reader of that observation:

* `record_query_length()` is called at the one KB call site that encodes a query
  (`kb_rag.query`). It appends one JSON line per query, and the count comes from
  `colbert_encoder.count_tokens()`. That is an UNTRUNCATED, unpadded tokenization.
  A count read from `encode()`'s own output is always exactly the cap, so it would
  measure the cap rather than the query.
* `summarize()` turns those lines into p50 / p95 / max / over-cap rate, per
  (encoder, cap) group, because token counts from different tokenizers are not
  comparable.
* `belief_rows()` projects each group into `ClaimTuple`-shaped rows for the belief
  kernel (root `scripts/vidya/adapters/kb_rag_query_length.py`). The rows cite no
  protocol, so every one is an OBSERVATION, never decision-gating. That is a true
  statement about traffic telemetry, and nothing here grades.

Records carry the query's sha256 and character length, never its text.

The log goes to `KB_RAG_QUERY_LENGTH_LOG`, which defaults to
`data/kb_rag/telemetry/query_lengths.jsonl` (gitignored). Set it to `0`, `off` or
`false` to disable the log. Instrumentation failures are logged at debug level and
never affect retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SCHEMA = "epyc.kb_rag.query_length.v1"
REPORT_SCHEMA = "epyc.kb_rag.query_length_report.v1"
BELIEF_SCHEMA = "epyc.kb_rag.query_length_belief.v1"
PRODUCER_ID = "epyc-orchestrator:src/retrieval/kb_rag_query_telemetry.py"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = _REPO_ROOT / "data" / "kb_rag" / "telemetry" / "query_lengths.jsonl"
LOG_ENV = "KB_RAG_QUERY_LENGTH_LOG"
_DISABLED = {"0", "off", "false", "no", "disabled"}


def log_path() -> Path | None:
    """Resolved log path, or None when the instrument is disabled."""
    raw = os.environ.get(LOG_ENV, "").strip()
    if raw.lower() in _DISABLED:
        return None
    return Path(raw).expanduser() if raw else DEFAULT_LOG_PATH


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_record(
    text: str,
    *,
    query_tokens: int,
    cap: int,
    role: str,
    prefix_convention: str,
    encoder_model_dir: str,
    encoder_slot: str,
    index_dir: str,
    ts: str | None = None,
) -> dict[str, Any]:
    """One query-length observation. `query_tokens` MUST be untruncated."""
    return {
        "schema": SCHEMA,
        "ts": ts or _utc_now(),
        "query_tokens": int(query_tokens),
        "cap": int(cap),
        "over_cap": int(query_tokens) > int(cap),
        "role": role,
        "prefix_convention": prefix_convention,
        "encoder_model_dir": encoder_model_dir,
        "encoder_slot": encoder_slot,
        "index_dir": index_dir,
        "query_chars": len(text),
        "query_sha256": hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest(),
        "count_basis": "untruncated_unpadded_tokenization_with_role_prefix_and_specials",
    }


def record_query_length(
    text: str,
    *,
    cap: int,
    role: str,
    prefix_convention: str,
    index_dir: str | Path,
) -> dict[str, Any] | None:
    """Count `text` untruncated and append one record. Never raises."""
    try:
        path = log_path()
        if path is None:
            return None
        from src.retrieval import colbert_encoder

        with colbert_encoder.locked_state():
            n = colbert_encoder.count_tokens(text, role=role)
            if n is None:
                return None  # no count is recorded rather than a fabricated one
            record = build_record(
                text,
                query_tokens=n,
                cap=cap,
                role=role,
                prefix_convention=prefix_convention,
                encoder_model_dir=str(colbert_encoder._MODEL_DIR),  # noqa: SLF001
                encoder_slot=str(colbert_encoder._MODEL_SLOT),  # noqa: SLF001
                index_dir=str(index_dir),
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        # One O_APPEND write per record, so concurrent writers never interleave a line.
        fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        if record["over_cap"]:
            logger.info("kb_rag: query is %d tokens, over the %d-token cap (truncated)", n, cap)
        return record
    except Exception as e:  # noqa: BLE001 — instrumentation must never fail a query
        logger.debug("kb_rag query-length record failed (%s): %s", type(e).__name__, e)
        return None


# ── reader ──────────────────────────────────────────────────────────────────


def read_log(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a log snapshot. Returns (records, snapshot identity).

    The snapshot is the exact byte prefix read. The log is append-only, so its sha256
    stays re-derivable later over the first `bytes` bytes, even after the file grows.
    """
    data = path.read_bytes()
    # Only whole lines count: a writer may be mid-append at the tail.
    cut = data.rfind(b"\n") + 1
    data = data[:cut]
    records: list[dict[str, Any]] = []
    malformed = 0
    foreign = 0
    for raw in data.splitlines():
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            malformed += 1
            continue
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
            foreign += 1
            continue
        tokens, cap = rec.get("query_tokens"), rec.get("cap")
        if not (isinstance(tokens, int) and isinstance(cap, int)) or isinstance(tokens, bool):
            malformed += 1
            continue
        records.append(rec)
    snapshot = {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "malformed_lines": malformed,
        "foreign_lines": foreign,
    }
    return records, snapshot


def _nearest_rank(sorted_values: list[int], pct: float) -> int:
    """Nearest-rank percentile, which is always an observed value."""
    k = max(1, math.ceil(pct / 100.0 * len(sorted_values)))
    return sorted_values[k - 1]


def summarize(records: Iterable[dict[str, Any]], *, since: str | None = None) -> list[dict[str, Any]]:
    """p50/p95/max/over-cap rate per (encoder_model_dir, cap, prefix_convention) group.

    A group only exists if it has at least one observation. An empty log yields no
    groups, never a 0 % rate: absence of traffic is not absence of truncation.
    """
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for rec in records:
        if since and str(rec.get("ts", "")) < since:
            continue
        key = (str(rec.get("encoder_model_dir", "")), int(rec["cap"]),
               str(rec.get("prefix_convention", "")))
        groups.setdefault(key, []).append(rec)
    out = []
    for (encoder, cap, convention), recs in sorted(groups.items()):
        values = sorted(int(r["query_tokens"]) for r in recs)
        over = sum(1 for v in values if v > cap)
        stamps = sorted(str(r.get("ts", "")) for r in recs)
        out.append({
            "encoder_model_dir": encoder,
            "cap": cap,
            "prefix_convention": convention,
            "n": len(values),
            "p50": _nearest_rank(values, 50),
            "p95": _nearest_rank(values, 95),
            "max": values[-1],
            "over_cap_count": over,
            "over_cap_rate": over / len(values),
            "first_ts": stamps[0],
            "last_ts": stamps[-1],
        })
    return out


def build_report(path: Path, *, since: str | None = None) -> dict[str, Any]:
    records, snapshot = read_log(path)
    groups = summarize(records, since=since)
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": _utc_now(),
        "since": since,
        "log": snapshot,
        "observations": sum(g["n"] for g in groups),
        "groups": groups,
        "belief_measurements": belief_rows(groups, snapshot),
    }


_METRICS = (
    # (key, metric name, unit, how to render)
    ("over_cap_rate", "kb_rag.query_over_cap_rate", "fraction", "{:.4f}"),
    ("p50", "kb_rag.query_tokens_p50", "tokens", "{}"),
    ("p95", "kb_rag.query_tokens_p95", "tokens", "{}"),
    ("max", "kb_rag.query_tokens_max", "tokens", "{}"),
)


def belief_rows(groups: list[dict[str, Any]], snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """ClaimTuple field dicts, one per (group, metric). PROJECTION only, never a grade.

    * No `protocol_id`: no codified protocol covers traffic telemetry, so the kernel
      grades these as OBSERVATIONS. None is invented.
    * `metric_direction` is recorded by the producer: `lower_better` means more headroom
      under the truncation cap. The basis is carried in `extra`.
    * The attestation is the log path plus the exact byte range read. The prefix sha256
      is carried in `extra`, not in `attestation_sha256`, because the live file keeps
      growing and a whole-file digest claim would be false by the next query.
    """
    rows: list[dict[str, Any]] = []
    for g in groups:
        ident_src = f"{snapshot['sha256']}|{g['encoder_model_dir']}|{g['cap']}|{g['prefix_convention']}"
        group_id = hashlib.sha256(ident_src.encode()).hexdigest()[:16]
        for key, metric, unit, fmt in _METRICS:
            value = g[key]
            rendered = fmt.format(value)
            if key == "over_cap_rate":
                text = (f"{g['over_cap_count']} of {g['n']} KB-RAG queries ({rendered}) exceeded the "
                        f"{g['cap']}-token query cap")
            else:
                text = f"KB-RAG query length {key} = {rendered} tokens over {g['n']} queries (cap {g['cap']})"
            text += f" [{g['encoder_model_dir']}, {g['prefix_convention']}, {g['first_ts']}..{g['last_ts']}]"
            rows.append({
                "measurement_id": f"kbrag-qlen-{group_id}-{key}",
                "metric": metric,
                "value": value,
                "unit": unit,
                "date": g["last_ts"],
                "category": "BASELINE",
                "claim": text,
                "metric_direction": "lower_better",
                "protocol_id": "",
                "reps": g["n"],
                "reps_basis": "scored: queries with a recorded untruncated token count",
                "attestation_path": snapshot["path"],
                "attestation_locator": f"{snapshot['path']}#bytes=0-{snapshot['bytes']}",
                "source_kind": "measurement",
                "extra": {
                    "belief_schema": BELIEF_SCHEMA,
                    "producer": PRODUCER_ID,
                    "log_prefix_sha256": snapshot["sha256"],
                    "log_prefix_bytes": snapshot["bytes"],
                    "cap": g["cap"],
                    "encoder_model_dir": g["encoder_model_dir"],
                    "prefix_convention": g["prefix_convention"],
                    "window": [g["first_ts"], g["last_ts"]],
                    "over_cap_count": g["over_cap_count"],
                    "direction_basis": "lower = more headroom under the query truncation cap",
                    "percentile_method": "nearest-rank",
                },
            })
    return rows
