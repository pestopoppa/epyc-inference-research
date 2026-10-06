"""Prospective completed-writer catalog dependency evidence, never a belief.

The named final commits in build_index/update_files/remove_files, parallel
prepare and parallel rebuild publish a native row in the SAME transaction as
that commit. Schema/FTS initialization, build's every-25-file checkpoints and
post-rebuild VACUUM are not completion records. Earlier commits cannot be rolled
back by a later failure. Hashing costs O(logical catalog content) at each named
boundary; no timing, vector-byte/existence or retrieval-quality claim is made.

Readers reopen producer-written rows only, with mode=ro and a read transaction.
Legacy absence stays unknown. There is no model import, historical reconstruction,
ClaimTuple, ledger enrollment or grader in this module.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

TABLE = 'catalog_dependency'
SCHEMA = (
    'CREATE TABLE catalog_dependency ('
    'singleton INTEGER PRIMARY KEY CHECK(singleton = 1), '
    'record_json TEXT NOT NULL, record_sha256 TEXT NOT NULL)'
)
FORMAT = 'epyc.kb.catalog-dependency.v1'
BASIS = 'ordered-logical-chunk-meta-fts-and-declared-schema.v1'
OPERATIONS = {'build_index', 'update_files', 'remove_files',
              'parallel_prepare', 'parallel_rebuild'}
STORED_KEYS = ('tokenizer_sha256', 'embedding_dim', 'query_max_tokens',
               'doc_max_tokens', 'prefix_convention', 'encoder_model_dir',
               'encoder_model_file', 'stamped_at')
LOADED_KEYS = {'tokenizer_sha256', 'embedding_dim', 'model_dir', 'model_slot', 'generation'}


def _bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _hash(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _validate_loaded(loaded: dict | None) -> None:
    if loaded is None:
        return
    if not isinstance(loaded, dict) or set(loaded) != LOADED_KEYS:
        raise ValueError('undeclared loaded identity fields')
    digest = loaded['tokenizer_sha256']
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                               or any(ch not in '0123456789abcdef' for ch in digest)):
        raise ValueError('invalid loaded tokenizer digest')
    width = loaded['embedding_dim']
    if width is not None and (type(width) is not int or width <= 0):
        raise ValueError('invalid loaded embedding width')
    if type(loaded['generation']) is not int or loaded['generation'] < 0:
        raise ValueError('invalid loaded generation')
    if any(not isinstance(loaded[key], str) for key in ('model_dir', 'model_slot')):
        raise ValueError('invalid loaded model identity')


def _stored(conn: sqlite3.Connection) -> dict:
    meta = dict(conn.execute('SELECT key, value FROM index_meta'))
    # Missing/empty legacy stamps stay unknown, including absent cap keys.
    return {key: meta.get(key) or None for key in STORED_KEYS}


def logical_digest(conn: sqlite3.Connection) -> str:
    """Stream only explicitly named logical content; exclude own table/schema."""
    digest = hashlib.sha256()
    digest.update(_bytes(BASIS) + b'\n')
    for name in ('chunk', 'index_meta', 'chunk_fts', 'chunk_file', 'chunk_hash'):
        schema = conn.execute('SELECT type, name, sql FROM sqlite_master WHERE name=?',
                              (name,)).fetchone()
        digest.update(_bytes([name, list(schema) if schema else None]) + b'\n')
    for name, query in (
        ('chunk', 'SELECT * FROM chunk ORDER BY chunk_id'),
        ('index_meta', 'SELECT key, value FROM index_meta ORDER BY key'),
        ('chunk_fts', 'SELECT rowid, file_path, heading_path, text FROM chunk_fts ORDER BY rowid'),
    ):
        exists = conn.execute('SELECT 1 FROM sqlite_master WHERE name=?', (name,)).fetchone()
        if not exists:
            if name != 'chunk_fts':
                raise ValueError(f'catalog lacks required logical table {name}')
            continue
        for row in conn.execute(query):
            digest.update(_bytes([name, list(row)]) + b'\n')
    return digest.hexdigest()


def _check_table(conn: sqlite3.Connection) -> bool:
    objects = list(conn.execute('SELECT type, name, sql FROM sqlite_master WHERE tbl_name=?',
                                (TABLE,)))
    if not objects:
        return False
    if [tuple(row) for row in objects] != [('table', TABLE, SCHEMA)]:
        raise ValueError('catalog dependency table/schema changed')
    return True


@contextmanager
def writer_connection(conn: sqlite3.Connection):
    """Close every writer path and roll back its currently active transaction."""
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def commit_completed_writer(conn: sqlite3.Connection, operation: str, *, loaded: dict | None) -> dict:
    """Publish completion identity and commit; caller owns cleanup/rollback scope."""
    if operation not in OPERATIONS:
        raise ValueError('undeclared completed-writer boundary')
    _validate_loaded(loaded)
    if not conn.in_transaction:
        conn.execute('BEGIN')
    if not _check_table(conn):
        conn.execute(SCHEMA)
    record = {
        'format': FORMAT,
        'digest_basis': BASIS,
        'operation': operation,
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'logical_catalog_sha256': logical_digest(conn),
        'stored_identity': _stored(conn),
        'loaded_identity': loaded,
        'scope': 'named completed-writer final transaction; logical catalog only',
    }
    body = _bytes(record)
    conn.execute('INSERT INTO catalog_dependency VALUES (1, ?, ?) '
                 'ON CONFLICT(singleton) DO UPDATE SET record_json=excluded.record_json, '
                 'record_sha256=excluded.record_sha256', (body.decode('utf-8'), _hash(body)))
    conn.commit()
    return record


def read_dependency(catalog: Path | str) -> dict | None:
    """Return an original current native row, or unknown; never create a file/row."""
    path = Path(catalog).resolve()
    if not path.is_file():
        return None
    conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        conn.execute('BEGIN')  # one read snapshot for native row + logical content
        if not _check_table(conn):
            return None
        rows = list(conn.execute('SELECT singleton, record_json, record_sha256 FROM catalog_dependency'))
        if len(rows) != 1 or rows[0][0] != 1:
            raise ValueError('catalog dependency must contain exactly one native row')
        _, body, seal = rows[0]
        if not isinstance(body, str) or _hash(body.encode('utf-8')) != seal:
            raise ValueError('catalog dependency record bytes changed')
        record = json.loads(body)
        if not isinstance(record, dict):
            raise ValueError("catalog dependency native JSON must be an object")
        fields = {'format', 'digest_basis', 'operation', 'captured_at',
                  'logical_catalog_sha256', 'stored_identity', 'loaded_identity', 'scope'}
        if (set(record) != fields or record['format'] != FORMAT or record['digest_basis'] != BASIS
                or record['operation'] not in OPERATIONS
                or record['scope'] != 'named completed-writer final transaction; logical catalog only'
                or _bytes(record).decode('utf-8') != body):
            raise ValueError('catalog dependency record contract changed')
        _validate_loaded(record['loaded_identity'])
        captured = datetime.fromisoformat(record['captured_at'])
        if captured.utcoffset() != timezone.utc.utcoffset(captured):
            raise ValueError('catalog dependency capture time must be UTC')
        if (record['logical_catalog_sha256'] != logical_digest(conn)
                or record['stored_identity'] != _stored(conn)):
            raise ValueError('catalog content no longer matches native dependency')
        return record
    finally:
        conn.close()


def export_dependency(catalog: Path | str) -> bytes | None:
    """Model-free bytes for existing document/artifact custody; no inferred row."""
    record = read_dependency(catalog)
    return None if record is None else _bytes(record) + b'\n'
