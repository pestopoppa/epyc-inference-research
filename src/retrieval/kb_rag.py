"""Internal Knowledge-Base RAG over the project's markdown corpus.

Architecture:
- Corpus: wiki/, handoffs/active/, handoffs/completed/, research/, progress/,
  + cross-repo docs/chapters/. Configured via kb_rag_config.yaml.
- Chunker: heading-aware (markdown_chunker.py).
- Encoder: shared ColBERT primitives (colbert_encoder.py).
- Storage: per-document .npz of token embeddings + SQLite catalog mapping
  (chunk_id, file_path, heading_path, line_range, mtime, content_hash), plus an
  `index_meta` table recording which encoder + prefix convention produced them.
- Query: top-K MaxSim ranking, returns chunk dicts with breadcrumbs.

Prefix convention (OP-24, 2026-08-12)
-------------------------------------
Stored vectors and query vectors must come from the SAME ColBERT role
convention; mixing them is worse than using the wrong one consistently. So the
INDEX is authoritative: `index_meta.prefix_convention` decides the role passed
to `colbert_encoder.encode()` for both sides, and every writer honours it.

Consequences, by design:
- A pre-OP-24 index has no `index_meta` row. It is read as
  `LEGACY_CONVENTION` ("none") and keeps serving prefix-free queries against
  its prefix-free vectors — self-consistent, exactly as before.
- A convention is adopted only by an EMPTY index directory. `--force` re-encodes
  but never re-stamps, so a live index can never be half-migrated in place.
- Migrating therefore means building a fresh index dir and pointing readers at
  it; the previous directory is left untouched and is the rollback.

Per handoffs/active/internal-kb-rag.md K3+K4.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import sqlite3
import stat
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.retrieval import colbert_encoder, cross_encoder, kb_catalog_dependency, kb_rag_query_telemetry
from src.retrieval.markdown_chunker import chunk_file

logger = logging.getLogger(__name__)

# Default index location under data/kb_rag/ (gitignored).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_KB_RAG_DATA = _REPO_ROOT / "data" / "kb_rag"

# Pre-OP-24 store: prefix-free vectors. Retained on disk as the rollback for the
# 2026-08-12 [Q]/[D] migration; still readable (see the module docstring).
LEGACY_INDEX_DIR = _KB_RAG_DATA / "index"

# Live store. `KB_RAG_INDEX_DIR` overrides it, which is the no-code-change
# rollback lever: point it at LEGACY_INDEX_DIR to serve the old vectors.
DEFAULT_INDEX_DIR = Path(
    os.environ.get("KB_RAG_INDEX_DIR") or (_KB_RAG_DATA / "index-qd-v1")
)

# Encoding bounds.
_QUERY_MAX_TOKENS = 48
_DOC_MAX_TOKENS = 256  # higher than reranker's 64; chunks are markdown sections


def _index_token_caps(meta: dict[str, str]) -> tuple[int, int]:
    """Stored caps are authoritative; missing historical keys keep old bounds."""
    caps = []
    for key, fallback in (
        ("query_max_tokens", _QUERY_MAX_TOKENS),
        ("doc_max_tokens", _DOC_MAX_TOKENS),
    ):
        value = meta.get(key)
        if value is None:
            caps.append(fallback)
        elif not value.isascii() or not value.isdecimal() or int(value) <= 0:
            raise RuntimeError(f"kb_rag: invalid stored {key}: {value!r}")
        else:
            caps.append(int(value))
    return caps[0], caps[1]


def _read_index_meta(index_dir: Path) -> dict[str, str]:
    """Read existing metadata without creating a catalog or running DDL."""
    catalog = index_dir / "catalog.sqlite"
    if not catalog.exists():
        return {}
    with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        meta = _read_meta(conn)
        _index_token_caps(meta)
        return meta


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# K10 (temporal recency) + K9 (cross-encoder rerank) hybrid-signal defaults.
# All default to a no-op so query() behaviour is unchanged until the K7 eval
# harness tunes them; env vars let the harness sweep without code edits.
# Source: agentmemory deep-dive, research/deep-dives/2026-05-27-agent-memory-cluster.md.
_RECENCY_WEIGHT_DEFAULT = _env_float("KB_RAG_RECENCY_WEIGHT", 0.0)  # 0 = MaxSim-only
_RECENCY_SIGMA_DAYS_DEFAULT = _env_float("KB_RAG_RECENCY_SIGMA_DAYS", 90.0)
_RERANK_DEFAULT = _env_flag("KB_RAG_RERANK")
_RERANK_WEIGHT_DEFAULT = _env_float("KB_RAG_RERANK_WEIGHT", 0.3)
_RERANK_POOL_MULT_DEFAULT = _env_float("KB_RAG_RERANK_POOL_MULT", 4.0)
_LEXICAL_WEIGHT_DEFAULT = _env_float("KB_RAG_LEXICAL_WEIGHT", 0.0)
_LEXICAL_POOL_MULT_DEFAULT = _env_float("KB_RAG_LEXICAL_POOL_MULT", 4.0)
_EMB_CACHE_DEFAULT = _env_flag("KB_RAG_EMB_CACHE")
_EMB_CACHE: dict[str, tuple[int, int, np.ndarray]] = {}


def _clear_embedding_cache() -> None:
    _EMB_CACHE.clear()


def _load_embedding(emb_path: Path) -> np.ndarray:
    if not _EMB_CACHE_DEFAULT:
        with np.load(emb_path) as data:
            return data["emb"]

    st = emb_path.stat()
    key = str(emb_path.resolve(strict=False))
    cached = _EMB_CACHE.get(key)
    if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]

    with np.load(emb_path) as data:
        emb = np.asarray(data["emb"]).copy()
    _EMB_CACHE[key] = (st.st_mtime_ns, st.st_size, emb)
    return emb


def _recency_score(mtime: float, now: float, sigma_days: float) -> float:
    """Gaussian recency weight in (0, 1]: 1.0 at age 0, decaying with age.

    age_days = (now - mtime) / 86400; score = exp(-0.5 * (age/sigma)^2).
    Matches agentmemory's temporal-proximity signal (intake-611).
    """
    if sigma_days <= 0:
        return 1.0
    age_days = max(0.0, (now - mtime) / 86400.0)
    return math.exp(-0.5 * (age_days / sigma_days) ** 2)


def _sanitize_fts_query(text: str) -> str:
    """Normalize a free-form query into an FTS5-safe token string."""
    return " ".join(
        tok for tok in "".join(c if c.isalnum() else " " for c in text).split() if tok
    )

_CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk (
  chunk_id INTEGER PRIMARY KEY,
  file_path TEXT NOT NULL,
  heading_path TEXT NOT NULL,        -- JSON-encoded list[str]
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  mtime REAL NOT NULL,
  emb_path TEXT NOT NULL,            -- relative to index_dir
  text_preview TEXT,                  -- first 240 chars for snippet display
  token_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS chunk_file ON chunk(file_path);
CREATE INDEX IF NOT EXISTS chunk_hash ON chunk(content_hash);

-- Provenance of the stored vectors. A store that cannot name the encoder and
-- prefix convention that produced it cannot be validated against a reader.
CREATE TABLE IF NOT EXISTS index_meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""


def _read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    """Return the index_meta key/value map (empty for a pre-OP-24 catalog)."""
    try:
        rows = conn.execute("SELECT key, value FROM index_meta").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(r[0]): str(r[1]) for r in rows}


def _index_convention(conn: sqlite3.Connection) -> str | None:
    """Prefix convention this index was built under.

    Returns None only for an EMPTY, unstamped catalog — the one state in which a
    writer may adopt the encoder's current convention. A populated catalog with
    no stamp predates OP-24 and is therefore prefix-free.
    """
    meta = _read_meta(conn)
    stamped = meta.get("prefix_convention")
    if stamped:
        return stamped
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    return colbert_encoder.LEGACY_CONVENTION if n_chunks else None


def _encoder_embedding_dim() -> int | None:
    """Embedding width the loaded ONNX graph emits, or None if unknowable.

    Taken from the graph's own output shape rather than a config file, because
    the graph is what produces the vectors. Returns None on a symbolic or
    missing dimension; callers must treat an unknown width as "cannot check",
    never as "matches".
    """
    session = getattr(colbert_encoder, "_session", None)
    if session is None:
        return None
    try:
        dim = session.get_outputs()[0].shape[-1]
    except Exception:  # noqa: BLE001
        return None
    return dim if isinstance(dim, int) and dim > 0 else None


@colbert_encoder.state_transaction
def _catalog_loaded_identity() -> dict | None:
    """Snapshot only already-loaded native identity; never initialize a model."""
    if colbert_encoder._session is None or colbert_encoder._tokenizer is None:
        return None
    return {
        "tokenizer_sha256": colbert_encoder._tokenizer_sha256,
        "embedding_dim": _encoder_embedding_dim(),
        "model_dir": str(colbert_encoder._MODEL_DIR),
        "model_slot": colbert_encoder._MODEL_SLOT,
        "generation": colbert_encoder.generation(),
    }


def _check_embedding_dim(meta: dict[str, str]) -> None:
    """Refuse a live encoder whose width differs from the one that built the index.

    Loud on purpose (K9). The alternative is a numpy broadcast error inside
    maxsim(), caught by encode()'s broad `except`, logged at debug, and returned
    as "no results" -- indistinguishable from an empty corpus.
    """
    stored = (meta.get("embedding_dim") or "").strip()
    if not stored:
        return  # index predates the stamp; nothing to compare against
    live = _encoder_embedding_dim()
    if live is None or str(live) == stored:
        return
    raise RuntimeError(
        f"kb_rag: index was built with {stored}-dim embeddings but the loaded "
        f"encoder emits {live}-dim ({colbert_encoder._MODEL_DIR}). MaxSim across "
        f"different widths is not a comparison; rebuild the index or restore the "
        f"original encoder."
    )


def _check_tokenizer_identity(meta: dict[str, str]) -> None:
    """Refuse known tokenizer drift; leave unstamped legacy stores unchanged."""
    stored = meta.get("tokenizer_sha256")
    if not stored:
        return
    loaded = colbert_encoder._tokenizer_sha256
    if loaded != stored:
        raise RuntimeError(
            "kb_rag: loaded tokenizer identity differs from the index stamp "
            "or is unknown; restore the original tokenizer or build a fresh index."
        )
    try:
        current = hashlib.sha256(colbert_encoder._TOKENIZER_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError("kb_rag: cannot verify stamped tokenizer file identity") from exc
    if current != loaded:
        raise RuntimeError(
            "kb_rag: tokenizer file changed since the encoder was loaded; "
            "restore the original tokenizer or build a fresh index."
        )


@colbert_encoder.state_transaction
def _stamp_meta(conn: sqlite3.Connection, convention: str) -> None:
    """Record encoder identity + convention alongside the vectors."""
    _check_tokenizer_identity(
        {"tokenizer_sha256": colbert_encoder._tokenizer_sha256 or ""}
    )
    values = {
        "prefix_convention": convention,
        "query_prefix": colbert_encoder.prefix_for_role(colbert_encoder.ROLE_QUERY)
        if convention == colbert_encoder.PREFIX_CONVENTION
        else "",
        "document_prefix": colbert_encoder.prefix_for_role(colbert_encoder.ROLE_DOCUMENT)
        if convention == colbert_encoder.PREFIX_CONVENTION
        else "",
        "encoder_model_dir": str(colbert_encoder._MODEL_DIR),
        "encoder_model_file": colbert_encoder._MODEL_PATH.name,
        "doc_max_tokens": str(colbert_encoder.max_document_tokens()),
        "query_max_tokens": str(colbert_encoder.max_query_tokens()),
        # K9: the identity that actually breaks the arithmetic. A model swap
        # changing embedding WIDTH (128 -> 64 across the mxbai family) leaves
        # every key above looking plausible; the mismatch then surfaces as a
        # numpy shape error inside maxsim(), swallowed by the broad `except` in
        # encode() and reported as an ordinary miss. Stamping the width lets a
        # reader refuse before scoring instead of scoring nonsense.
        "embedding_dim": str(_encoder_embedding_dim() or ""),
        "tokenizer_sha256": colbert_encoder._tokenizer_sha256 or "",
        "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    conn.executemany(
        "INSERT INTO index_meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        list(values.items()),
    )


def _roles_for_convention(convention: str) -> tuple[str, str]:
    """Map a stored convention to the (query_role, document_role) to encode with."""
    if convention == colbert_encoder.PREFIX_CONVENTION:
        return colbert_encoder.ROLE_QUERY, colbert_encoder.ROLE_DOCUMENT
    if convention == colbert_encoder.LEGACY_CONVENTION:
        return colbert_encoder.ROLE_NONE, colbert_encoder.ROLE_NONE
    raise ValueError(
        f"index was built under unknown prefix convention {convention!r}; "
        f"this code understands {colbert_encoder.PREFIX_CONVENTION!r} and "
        f"{colbert_encoder.LEGACY_CONVENTION!r}"
    )


def _writer_convention(conn: sqlite3.Connection) -> str:
    """Convention a writer must use, adopting + stamping only on an empty index."""
    meta = _read_meta(conn)
    _index_token_caps(meta)
    _check_tokenizer_identity(meta)
    convention = _index_convention(conn)
    if convention is None and meta:
        # Partial historical metadata is still identity; never replace stored
        # cap keys merely because an empty catalog lacks a convention stamp.
        convention = colbert_encoder.LEGACY_CONVENTION
    if convention is None:
        convention = colbert_encoder.PREFIX_CONVENTION
        _stamp_meta(conn, convention)
        logger.info("kb_rag: new index stamped with prefix convention %s", convention)
    elif convention == colbert_encoder.LEGACY_CONVENTION:
        logger.warning(
            "kb_rag: writing into a pre-OP-24 prefix-free index; new chunks stay "
            "prefix-free for consistency. Build a fresh index dir to migrate."
        )
    return convention


@colbert_encoder.state_transaction
def _warn_on_encoder_drift(meta: dict[str, str]) -> None:
    """Log if the live encoder is not the one that produced the stored vectors."""
    stored_dir = meta.get("encoder_model_dir")
    if stored_dir and stored_dir != str(colbert_encoder._MODEL_DIR):
        logger.warning(
            "kb_rag: index was built with encoder %s but %s is loaded — "
            "MaxSim scores across different models are not comparable",
            stored_dir, colbert_encoder._MODEL_DIR,
        )
    # A different DIRECTORY is a warning because the scores may still be
    # meaningful; a different WIDTH is not comparable at all, so it raises.
    _check_embedding_dim(meta)
    _check_tokenizer_identity(meta)


@dataclass
class CorpusConfig:
    """Roots + glob patterns for corpus walking."""

    roots: list[str]
    include_globs: list[str]
    exclude_patterns: list[str]
    max_chunk_chars: int = 4000

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CorpusConfig":
        import yaml  # lazy import

        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(
            roots=d.get("roots", []),
            include_globs=d.get("include_globs", ["**/*.md"]),
            exclude_patterns=d.get("exclude_patterns", []),
            max_chunk_chars=int(d.get("max_chunk_chars", 4000)),
        )


def _ensure_catalog(
    index_dir: Path, *, verify_tokenizer: bool = False,
) -> sqlite3.Connection:
    index_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_dir / "catalog.sqlite"))
    try:
        if verify_tokenizer:
            meta = _read_meta(conn)
            _index_token_caps(meta)
            _check_tokenizer_identity(meta)
    except RuntimeError:
        conn.close()
        raise
    conn.executescript(_CATALOG_SCHEMA)
    conn.commit()
    return conn


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    """Create the optional lexical index if SQLite was built with FTS5."""
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
            USING fts5(file_path, heading_path, text, tokenize='porter')
            """
        )
        conn.commit()
        return True
    except sqlite3.OperationalError:
        logger.info("FTS5 not available — KB-RAG lexical signal disabled")
        return False


def _sync_chunk_fts_row(
    cur: sqlite3.Cursor,
    chunk_id: int,
    file_path: str,
    heading_path: list[str],
    text: str,
    fts_enabled: bool,
) -> None:
    if not fts_enabled:
        return
    cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
    cur.execute(
        "INSERT INTO chunk_fts(rowid, file_path, heading_path, text) VALUES (?, ?, ?, ?)",
        (chunk_id, file_path, json.dumps(heading_path), text),
    )


def _lexical_scores(
    conn: sqlite3.Connection,
    text: str,
    limit: int,
) -> dict[int, float]:
    """Return reciprocal-rank lexical scores from FTS5 matches."""
    query = _sanitize_fts_query(text)
    if not query:
        return {}
    try:
        rows = conn.execute(
            "SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? "
            "ORDER BY bm25(chunk_fts) LIMIT ?",
            (query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}
    return {int(row[0]): 1.0 / float(idx + 1) for idx, row in enumerate(rows)}


def _walk_corpus(config: CorpusConfig) -> list[Path]:
    """Resolve all corpus markdown files per the config."""
    seen: set[Path] = set()
    files: list[Path] = []
    for root in config.roots:
        rp = Path(root).expanduser().resolve()
        if not rp.exists():
            logger.warning("corpus root %s does not exist — skipping", rp)
            continue
        for pattern in config.include_globs:
            for f in rp.glob(pattern):
                if not f.is_file():
                    continue
                # Apply exclude patterns (substring match against absolute path).
                rel = str(f)
                if any(ex in rel for ex in config.exclude_patterns):
                    continue
                if f in seen:
                    continue
                seen.add(f)
                files.append(f)
    return sorted(files)


def _emb_relative_path(file_path: str, content_hash: str) -> str:
    """Path of the per-chunk .npz under index_dir (relative)."""
    safe_name = Path(file_path).name.replace(".", "_")
    return f"emb/{safe_name}__{content_hash}.npz"


def _write_embedding_atomic(path: Path, emb: np.ndarray) -> None:
    """Publish a fully serialized vector without exposing a partial target."""
    target = path.resolve(strict=False)
    staged_path: Path | None = None
    staged_fd: int | None = None
    try:
        try:
            target_stat = target.lstat()
        except FileNotFoundError:
            target_mode = None
        else:
            if not stat.S_ISREG(target_stat.st_mode):
                raise OSError(f"embedding target is not a regular file: {target}")
            target_mode = stat.S_IMODE(target_stat.st_mode)

        for _ in range(10):
            candidate = target.with_name(f".{target.name}.{secrets.token_hex(12)}.tmp.npz")
            try:
                staged_fd = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o666,
                )
            except FileExistsError:
                continue
            staged_path = candidate
            break
        if staged_fd is None or staged_path is None:
            raise FileExistsError(f"could not allocate staged embedding beside {target}")
        if target_mode is not None:
            os.fchmod(staged_fd, target_mode)
        with os.fdopen(staged_fd, "wb") as staged:
            staged_fd = None
            np.savez_compressed(staged, emb=emb)
        os.replace(staged_path, target)
        staged_path = None
    except BaseException:
        if staged_fd is not None:
            try:
                os.close(staged_fd)
            except OSError:
                pass
        if staged_path is not None:
            try:
                staged_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("kb_rag: failed to clean staged embedding %s: %s", staged_path, exc)
        raise


@colbert_encoder.state_transaction
def build_index(
    config: CorpusConfig,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Build (or refresh) the KB-RAG index.

    For each markdown file:
    1. Read + chunk (heading-aware).
    2. For each chunk, check if `(file_path, content_hash)` already in catalog.
       If yes and force=False, skip (mtime-stable + content-stable).
    3. Otherwise, encode chunk text, write per-chunk .npz, upsert catalog row.

    Returns stats dict.
    """
    index_dir = Path(index_dir)
    _read_index_meta(index_dir)
    (index_dir / "emb").mkdir(parents=True, exist_ok=True)

    if not colbert_encoder.is_available():
        return {
            "ok": False,
            "error": "encoder model not available on disk",
            "model_path": str(colbert_encoder._MODEL_DIR),
        }

    if not colbert_encoder.ensure_loaded():
        return {"ok": False, "error": "encoder failed to load"}

    conn = _ensure_catalog(index_dir, verify_tokenizer=True)
    with kb_catalog_dependency.writer_connection(conn):
        conn.row_factory = sqlite3.Row
        convention = _writer_convention(conn)
        _, doc_cap = _index_token_caps(_read_meta(conn))
        fts_enabled = _ensure_fts(conn)
        _, doc_role = _roles_for_convention(convention)
        cur = conn.cursor()

        files = _walk_corpus(config)
        n_files = len(files)
        n_chunks_seen = 0
        n_chunks_encoded = 0
        n_chunks_skipped = 0
        started = time.perf_counter()

        for file_idx, f in enumerate(files, start=1):
            try:
                chunks = chunk_file(f, max_chars=config.max_chunk_chars)
            except Exception as e:  # noqa: BLE001
                logger.warning("chunker failed on %s: %s", f, e)
                continue

            mtime = f.stat().st_mtime
            current_chunk_keys: set[tuple[str, int, int]] = set()
            for ch in chunks:
                n_chunks_seen += 1
                line_start, line_end = ch.line_range
                current_chunk_keys.add((ch.content_hash, line_start, line_end))
                existing_rows = cur.execute(
                    "SELECT chunk_id, content_hash FROM chunk "
                    "WHERE file_path=? AND content_hash=? AND line_start=? AND line_end=? "
                    "ORDER BY chunk_id",
                    (str(f), ch.content_hash, line_start, line_end),
                ).fetchall()
                existing = existing_rows[0] if existing_rows else None
                if existing and not force:
                    n_chunks_skipped += 1
                    _sync_chunk_fts_row(
                        cur,
                        int(existing["chunk_id"]),
                        str(f),
                        ch.heading_path,
                        ch.text,
                        fts_enabled,
                    )
                    continue

                emb = colbert_encoder.encode(ch.text, doc_cap, role=doc_role)
                if emb is None:
                    continue

                emb_rel = _emb_relative_path(str(f), ch.content_hash)
                emb_abs = index_dir / emb_rel
                emb_abs.parent.mkdir(parents=True, exist_ok=True)
                _write_embedding_atomic(emb_abs, emb)

                preview = ch.text.strip()[:240]
                row_values = (
                    str(f),
                    json.dumps(ch.heading_path),
                    line_start,
                    line_end,
                    ch.content_hash,
                    mtime,
                    emb_rel,
                    preview,
                    int(emb.shape[0]),
                )
                if existing:
                    chunk_id = int(existing["chunk_id"])
                    cur.execute(
                        "UPDATE chunk SET file_path=?, heading_path=?, line_start=?, line_end=?, "
                        "content_hash=?, mtime=?, emb_path=?, text_preview=?, token_count=? "
                        "WHERE chunk_id=?",
                        (*row_values, chunk_id),
                    )
                    for duplicate in existing_rows[1:]:
                        duplicate_id = int(duplicate["chunk_id"])
                        if fts_enabled:
                            cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (duplicate_id,))
                        cur.execute("DELETE FROM chunk WHERE chunk_id = ?", (duplicate_id,))
                else:
                    cur.execute(
                        "INSERT INTO chunk "
                        "(file_path, heading_path, line_start, line_end, content_hash, "
                        " mtime, emb_path, text_preview, token_count) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        row_values,
                    )
                    chunk_id = int(cur.lastrowid)
                _sync_chunk_fts_row(
                    cur,
                    chunk_id,
                    str(f),
                    ch.heading_path,
                    ch.text,
                    fts_enabled,
                )
                n_chunks_encoded += 1

            stale_chunk_rows = cur.execute(
                "SELECT chunk_id, content_hash, line_start, line_end FROM chunk WHERE file_path = ?",
                (str(f),),
            ).fetchall()
            for row in stale_chunk_rows:
                key = (str(row["content_hash"]), int(row["line_start"]), int(row["line_end"]))
                if key in current_chunk_keys:
                    continue
                if fts_enabled:
                    cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (int(row["chunk_id"]),))
                cur.execute("DELETE FROM chunk WHERE chunk_id = ?", (int(row["chunk_id"]),))

            if file_idx % 25 == 0:
                conn.commit()
                logger.info(
                    "kb_rag: %d/%d files, %d chunks encoded, %d skipped",
                    file_idx, n_files, n_chunks_encoded, n_chunks_skipped,
                )

        # Optional cleanup: remove catalog rows for files that disappeared.
        catalog_files = {r[0] for r in cur.execute("SELECT DISTINCT file_path FROM chunk")}
        current_files = {str(f) for f in files}
        stale = catalog_files - current_files
        for stale_file in stale:
            if fts_enabled:
                for row in cur.execute(
                    "SELECT chunk_id FROM chunk WHERE file_path = ?",
                    (stale_file,),
                ).fetchall():
                    cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (int(row["chunk_id"]),))
            cur.execute("DELETE FROM chunk WHERE file_path = ?", (stale_file,))
        kb_catalog_dependency.commit_completed_writer(conn, "build_index", loaded=_catalog_loaded_identity())
    _clear_embedding_cache()

    elapsed = time.perf_counter() - started
    return {
        "ok": True,
        "files": n_files,
        "chunks_seen": n_chunks_seen,
        "chunks_encoded": n_chunks_encoded,
        "chunks_skipped_unchanged": n_chunks_skipped,
        "stale_files_removed": len(stale),
        "elapsed_sec": round(elapsed, 2),
        "index_dir": str(index_dir),
        "prefix_convention": convention,
    }


@colbert_encoder.state_transaction
def update_files(
    paths: list[str],
    config: CorpusConfig,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
) -> dict[str, Any]:
    """Re-encode a specific list of files (e.g. from `git diff --name-only`).

    Each file's existing catalog rows are deleted before re-chunking +
    re-encoding. Files not matching any include glob are skipped silently.
    """
    index_dir = Path(index_dir)
    _read_index_meta(index_dir)
    (index_dir / "emb").mkdir(parents=True, exist_ok=True)
    if not colbert_encoder.ensure_loaded():
        return {"ok": False, "error": "encoder failed to load"}

    conn = _ensure_catalog(index_dir, verify_tokenizer=True)
    with kb_catalog_dependency.writer_connection(conn):
        conn.row_factory = sqlite3.Row
        convention = _writer_convention(conn)
        _, doc_cap = _index_token_caps(_read_meta(conn))
        fts_enabled = _ensure_fts(conn)
        _, doc_role = _roles_for_convention(convention)
        cur = conn.cursor()

        # Resolve which paths actually belong to corpus.
        corpus_files = {str(p) for p in _walk_corpus(config)}
        encoded = 0
        for raw_path in paths:
            p = Path(raw_path).resolve()
            if str(p) not in corpus_files:
                continue
            if fts_enabled:
                for row in cur.execute(
                    "SELECT chunk_id, heading_path, line_start, line_end, text_preview "
                    "FROM chunk WHERE file_path = ?",
                    (str(p),),
                ).fetchall():
                    cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (int(row["chunk_id"]),))
            cur.execute("DELETE FROM chunk WHERE file_path = ?", (str(p),))
            chunks = chunk_file(p, max_chars=config.max_chunk_chars)
            mtime = p.stat().st_mtime
            for ch in chunks:
                emb = colbert_encoder.encode(ch.text, doc_cap, role=doc_role)
                if emb is None:
                    continue
                emb_rel = _emb_relative_path(str(p), ch.content_hash)
                emb_abs = index_dir / emb_rel
                emb_abs.parent.mkdir(parents=True, exist_ok=True)
                _write_embedding_atomic(emb_abs, emb)
                cur.execute(
                    "INSERT INTO chunk "
                    "(file_path, heading_path, line_start, line_end, content_hash, "
                    " mtime, emb_path, text_preview, token_count) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(p),
                        json.dumps(ch.heading_path),
                        ch.line_range[0],
                        ch.line_range[1],
                        ch.content_hash,
                        mtime,
                        emb_rel,
                        ch.text.strip()[:240],
                        int(emb.shape[0]),
                    ),
                )
                _sync_chunk_fts_row(
                    cur,
                    int(cur.lastrowid),
                    str(p),
                    ch.heading_path,
                    ch.text,
                    fts_enabled,
                )
                encoded += 1
        kb_catalog_dependency.commit_completed_writer(conn, "update_files", loaded=_catalog_loaded_identity())
    _clear_embedding_cache()
    return {
        "ok": True,
        "files_processed": len(paths),
        "chunks_encoded": encoded,
        "prefix_convention": convention,
    }


def remove_files(
    paths: list[str],
    index_dir: Path | str = DEFAULT_INDEX_DIR,
) -> dict[str, Any]:
    """Commit removed rows before model-free orphan-vector cleanup.

    Catalog/native-record failures preserve referenced bytes. Cleanup failures
    after commit are reported separately; they leave orphans, not live references
    to files deleted by a rolled-back transaction. Concurrent writers remain
    outside this per-call contract.
    """
    index_dir = Path(index_dir)
    conn = _ensure_catalog(index_dir)
    with kb_catalog_dependency.writer_connection(conn):
        conn.row_factory = sqlite3.Row
        fts_enabled = _ensure_fts(conn)
        cur = conn.cursor()

        removed_files = 0
        removed_chunks = 0
        pending_embeddings: set[str] = set()
        for raw_path in paths:
            p = Path(raw_path).expanduser().resolve()
            rows = cur.execute(
                "SELECT chunk_id, emb_path FROM chunk WHERE file_path = ?",
                (str(p),),
            ).fetchall()
            if not rows:
                continue
            removed_files += 1
            removed_chunks += len(rows)
            for row in rows:
                if fts_enabled:
                    cur.execute("DELETE FROM chunk_fts WHERE rowid = ?", (int(row["chunk_id"]),))
                pending_embeddings.add(str(row["emb_path"]))
            cur.execute("DELETE FROM chunk WHERE file_path = ?", (str(p),))

        # A vector shared by surviving rows is still active in this catalog.
        orphan_paths = [index_dir / relative for relative in sorted(pending_embeddings)
                        if not cur.execute("SELECT 1 FROM chunk WHERE emb_path=? LIMIT 1",
                                           (relative,)).fetchone()]
        kb_catalog_dependency.commit_completed_writer(conn, "remove_files", loaded=None)
    removed_embeddings = 0
    cleanup_errors = []
    for emb_path in orphan_paths:
        try:
            emb_path.unlink()
            removed_embeddings += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            cleanup_errors.append({"path": str(emb_path), "error": f"{type(exc).__name__}: {exc}"})
            logger.warning("kb_rag: committed removal left orphan %s: %s", emb_path, exc)
    _clear_embedding_cache()
    return {
        "ok": True,
        "files_removed": removed_files,
        "chunks_removed": removed_chunks,
        "embeddings_removed": removed_embeddings,
        "embedding_cleanup_ok": not cleanup_errors,
        "orphaned_embeddings": len(cleanup_errors),
        "embedding_cleanup_errors": cleanup_errors,
    }


@colbert_encoder.state_transaction
def query(
    text: str,
    top_k: int = 8,
    index_dir: Path | str = DEFAULT_INDEX_DIR,
    recency_weight: float | None = None,
    recency_sigma_days: float | None = None,
    rerank: bool | None = None,
    rerank_weight: float | None = None,
    lexical_weight: float | None = None,
) -> list[dict[str, Any]]:
    """Top-K hybrid retrieval against the indexed corpus.

    Stage 1 (always): ColBERT MaxSim over every chunk.
    Stage 2 (K10, opt-in): blend a Gaussian temporal-recency weight in
      `final = (1 - recency_weight) * maxsim + recency_weight * recency`.
    Stage 3 (K9, opt-in): cross-encoder rerank of the top `top_k * pool_mult`
      candidates, blended `(1 - rerank_weight) * score + rerank_weight * ce`.
    Stage 4 (K11, opt-in): blend a lexical FTS5 signal over the chunk text,
      file path, and heading path. Default weight is zero, so the path remains
      MaxSim-only unless a caller explicitly opts in.

    All optional stages default to a no-op so the default result is identical
    to plain MaxSim; defaults come from env vars (KB_RAG_RECENCY_WEIGHT /
    _SIGMA_DAYS, KB_RAG_RERANK / _WEIGHT / _POOL_MULT, KB_RAG_LEXICAL_WEIGHT
    / _POOL_MULT) so the K7 eval harness can sweep without code edits. Tuning +
    win-validation is gated on K7 (inference). See
    research/deep-dives/2026-05-27-agent-memory-cluster.md.

    Returns list of dicts: {file, heading_path, line_range, snippet, score}
    (plus `recency` and `ce_score` when those stages are active).
    """
    recency_weight = _RECENCY_WEIGHT_DEFAULT if recency_weight is None else recency_weight
    recency_sigma_days = (
        _RECENCY_SIGMA_DAYS_DEFAULT if recency_sigma_days is None else recency_sigma_days
    )
    rerank = _RERANK_DEFAULT if rerank is None else rerank
    rerank_weight = _RERANK_WEIGHT_DEFAULT if rerank_weight is None else rerank_weight
    lexical_weight = _LEXICAL_WEIGHT_DEFAULT if lexical_weight is None else lexical_weight

    index_dir = Path(index_dir)
    catalog_path = index_dir / "catalog.sqlite"
    if not catalog_path.exists():
        logger.warning("no index at %s — run build_index first", index_dir)
        return []

    if not colbert_encoder.ensure_loaded():
        return []

    conn = sqlite3.connect(str(catalog_path))
    conn.row_factory = sqlite3.Row

    # The index decides the convention for BOTH sides. Encoding the query under
    # a convention the stored vectors were not built with is worse than using
    # the legacy convention consistently, so this is never inferred locally.
    meta = _read_meta(conn)
    try:
        _warn_on_encoder_drift(meta)
        query_cap, _ = _index_token_caps(meta)
    except RuntimeError:
        conn.close()
        raise
    convention = _index_convention(conn) or colbert_encoder.PREFIX_CONVENTION
    try:
        query_role, _ = _roles_for_convention(convention)
    except ValueError as e:
        logger.error("kb_rag: %s", e)
        conn.close()
        return []

    q_emb = colbert_encoder.encode(text, query_cap, role=query_role)
    if q_emb is None:
        conn.close()
        return []
    # H2: observe the UNTRUNCATED query length (encode() output always equals the cap).
    kb_rag_query_telemetry.record_query_length(
        text,
        cap=query_cap,
        role=query_role,
        prefix_convention=convention,
        index_dir=index_dir,
    )

    rows = conn.execute(
        "SELECT chunk_id, file_path, heading_path, line_start, line_end, "
        "       content_hash, emb_path, text_preview, mtime FROM chunk"
    ).fetchall()

    lexical_scores = (
        _lexical_scores(conn, text, max(top_k, int(top_k * _LEXICAL_POOL_MULT_DEFAULT)))
        if lexical_weight > 0.0
        else {}
    )

    now = time.time()
    scored: list[tuple[float, float, float, sqlite3.Row]] = []  # (blended, recency, lexical, row)
    for r in rows:
        emb_path = index_dir / r["emb_path"]
        if not emb_path.exists():
            continue
        try:
            d_emb = _load_embedding(emb_path)
        except Exception:  # noqa: BLE001
            continue
        s = colbert_encoder.maxsim(q_emb, d_emb)
        rec = (
            _recency_score(r["mtime"], now, recency_sigma_days)
            if recency_weight > 0.0
            else 0.0
        )
        lexical = lexical_scores.get(int(r["chunk_id"]), 0.0)
        blended = s
        if recency_weight > 0.0:
            blended = (1.0 - recency_weight) * blended + recency_weight * rec
        if lexical_weight > 0.0:
            blended = (1.0 - lexical_weight) * blended + lexical_weight * lexical
        scored.append((blended, rec, lexical, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # K9: cross-encoder rerank a wider candidate pool, then truncate to top_k.
    pool_size = max(top_k, int(top_k * _RERANK_POOL_MULT_DEFAULT)) if rerank else top_k

    results = []
    for blended, rec, lexical, r in scored[:pool_size]:
        item = {
            "file": r["file_path"],
            "heading_path": json.loads(r["heading_path"]),
            "line_range": (r["line_start"], r["line_end"]),
            "snippet": r["text_preview"],
            "score": round(blended, 4),
            "content_hash": r["content_hash"],
        }
        if recency_weight > 0.0:
            item["recency"] = round(rec, 4)
        if lexical_weight > 0.0:
            item["lexical"] = round(lexical, 4)
        results.append(item)

    if rerank and rerank_weight > 0.0 and results:
        results = cross_encoder.rerank(
            text, results, text_key="snippet", weight=rerank_weight, base_key="score"
        )

    conn.close()
    return results[:top_k]


def stats(index_dir: Path | str = DEFAULT_INDEX_DIR) -> dict[str, Any]:
    """Index summary: chunk count, file count, total embedding bytes."""
    index_dir = Path(index_dir)
    catalog_path = index_dir / "catalog.sqlite"
    if not catalog_path.exists():
        return {"exists": False}
    conn = sqlite3.connect(str(catalog_path))
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunk").fetchone()[0]
    n_files = conn.execute("SELECT COUNT(DISTINCT file_path) FROM chunk").fetchone()[0]
    meta = _read_meta(conn)
    convention = _index_convention(conn)
    conn.close()
    emb_dir = index_dir / "emb"
    emb_bytes = sum(p.stat().st_size for p in emb_dir.glob("*.npz")) if emb_dir.exists() else 0
    return {
        "exists": True,
        "files": n_files,
        "chunks": n_chunks,
        "emb_bytes": emb_bytes,
        "emb_mib": round(emb_bytes / (1024 * 1024), 2),
        "index_dir": str(index_dir),
        "prefix_convention": convention,
        "meta": meta,
    }
