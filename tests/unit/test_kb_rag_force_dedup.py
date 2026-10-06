"""KB-RAG forced-build identity reconciliation using a fake encoder only."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.retrieval import colbert_encoder, kb_catalog_dependency, kb_rag


@pytest.fixture
def fake_encoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Mock:
    tokenizer = tmp_path / "fake-tokenizer.json"
    tokenizer.write_bytes(b"fixture bytes; never loaded as a tokenizer")
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    monkeypatch.setattr(colbert_encoder, "_TOKENIZER_PATH", tokenizer)
    monkeypatch.setattr(colbert_encoder, "_tokenizer_sha256", digest)
    monkeypatch.setattr(colbert_encoder, "_tokenizer", object())
    monkeypatch.setattr(
        colbert_encoder,
        "_session",
        SimpleNamespace(get_outputs=lambda: [SimpleNamespace(shape=[1, 1, 4])]),
    )
    monkeypatch.setattr(colbert_encoder, "_MODEL_DIR", tmp_path / "fake-model")
    monkeypatch.setattr(colbert_encoder, "_MODEL_PATH", tmp_path / "fake-model.onnx")
    monkeypatch.setattr(colbert_encoder, "_MODEL_SLOT", "fake-slot")
    monkeypatch.setattr(colbert_encoder, "_generation", 1)
    monkeypatch.setattr(colbert_encoder, "is_available", Mock(return_value=True))
    monkeypatch.setattr(colbert_encoder, "ensure_loaded", Mock(return_value=True))
    monkeypatch.setattr(colbert_encoder, "max_query_tokens", lambda: 32)
    monkeypatch.setattr(colbert_encoder, "max_document_tokens", lambda: 300)
    encode = Mock(return_value=np.full((2, 4), 0.25, dtype=np.float32))
    monkeypatch.setattr(colbert_encoder, "encode", encode)
    return encode


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[kb_rag.CorpusConfig, Path]:
    root = tmp_path / "corpus"
    root.mkdir()
    source = root / "one.md"
    source.write_text("# Fixture\n\nstable body text\n", encoding="utf-8")
    return (
        kb_rag.CorpusConfig([str(root)], ["*.md"], [], max_chunk_chars=4000),
        source,
    )


def _rows(index: Path) -> list[tuple]:
    with sqlite3.connect(index / "catalog.sqlite") as conn:
        return conn.execute(
            "SELECT chunk_id, file_path, content_hash, line_start, line_end, emb_path "
            "FROM chunk ORDER BY chunk_id"
        ).fetchall()


def _fts_rows(index: Path) -> list[tuple]:
    with sqlite3.connect(index / "catalog.sqlite") as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='chunk_fts'"
        ).fetchone(), "core force-dedup fixture requires SQLite FTS5"
        return conn.execute(
            "SELECT rowid, file_path, heading_path, text FROM chunk_fts ORDER BY rowid"
        ).fetchall()


def test_two_forced_builds_reuse_identity_and_publish_native_record(
    fake_encoder: Mock, corpus: tuple[kb_rag.CorpusConfig, Path], tmp_path: Path
) -> None:
    config, _ = corpus
    index = tmp_path / "index"

    first = kb_rag.build_index(config, index_dir=index, force=True)
    first_rows = _rows(index)
    second = kb_rag.build_index(config, index_dir=index, force=True)

    assert first["chunks_encoded"] == second["chunks_encoded"] == 1
    assert len(first_rows) == len(_rows(index)) == 1
    assert first_rows[0][0] == _rows(index)[0][0]
    fts = _fts_rows(index)
    assert len(fts) == 1
    assert fts[0][0] == first_rows[0][0]
    assert fake_encoder.call_count == 2
    record = kb_catalog_dependency.read_dependency(index / "catalog.sqlite")
    assert record is not None
    assert record["operation"] == "build_index"


def test_nonforced_unchanged_build_skips_encode_and_preserves_row(
    fake_encoder: Mock, corpus: tuple[kb_rag.CorpusConfig, Path], tmp_path: Path
) -> None:
    config, _ = corpus
    index = tmp_path / "index"
    kb_rag.build_index(config, index_dir=index, force=True)
    before = _rows(index)
    before_fts = _fts_rows(index)
    fake_encoder.reset_mock(side_effect=True, return_value=True)
    fake_encoder.side_effect = AssertionError("unchanged nonforced chunk was encoded")

    result = kb_rag.build_index(config, index_dir=index)

    assert result["chunks_skipped_unchanged"] == 1
    assert _rows(index) == before
    assert _fts_rows(index) == before_fts


def test_failed_forced_encode_preserves_existing_catalog_and_vector_bytes(
    fake_encoder: Mock, corpus: tuple[kb_rag.CorpusConfig, Path], tmp_path: Path
) -> None:
    config, _ = corpus
    index = tmp_path / "index"
    kb_rag.build_index(config, index_dir=index, force=True)
    before_rows = _rows(index)
    before_fts = _fts_rows(index)
    vector = index / before_rows[0][5]
    before_vector = vector.read_bytes()
    fake_encoder.side_effect = None
    fake_encoder.return_value = None

    result = kb_rag.build_index(config, index_dir=index, force=True)

    assert result["chunks_encoded"] == 0
    assert _rows(index) == before_rows
    assert _fts_rows(index) == before_fts
    assert vector.read_bytes() == before_vector
    assert kb_catalog_dependency.read_dependency(index / "catalog.sqlite") is not None


def test_changed_identity_prunes_old_row_without_deleting_its_vector(
    fake_encoder: Mock, corpus: tuple[kb_rag.CorpusConfig, Path], tmp_path: Path
) -> None:
    config, source = corpus
    index = tmp_path / "index"
    kb_rag.build_index(config, index_dir=index, force=True)
    before = _rows(index)
    old_vector = index / before[0][5]
    source.write_text("# Fixture\n\nchanged body identity\n", encoding="utf-8")

    kb_rag.build_index(config, index_dir=index, force=True)

    rows = _rows(index)
    assert len(rows) == 1
    assert rows[0][2] != before[0][2]
    assert [row[0] for row in _fts_rows(index)] == [rows[0][0]]
    assert old_vector.is_file()


def test_touched_duplicate_identity_converges_without_unlinking_active_vector(
    fake_encoder: Mock, corpus: tuple[kb_rag.CorpusConfig, Path], tmp_path: Path
) -> None:
    config, _ = corpus
    index = tmp_path / "index"
    other_source = Path(config.roots[0]) / "two.md"
    other_source.write_text("# Other\n\nunrelated body text\n", encoding="utf-8")
    kb_rag.build_index(config, index_dir=index, force=True)
    original = next(row for row in _rows(index) if row[1].endswith("one.md"))
    unrelated = next(row for row in _rows(index) if row[1].endswith("two.md"))
    active_vector = index / original[5]
    with sqlite3.connect(index / "catalog.sqlite") as conn:
        conn.execute("UPDATE chunk SET emb_path=? WHERE chunk_id=?", (original[5], unrelated[0]))
        conn.commit()
    unrelated_before = next(row for row in _rows(index) if row[0] == unrelated[0])
    unrelated_fts_before = next(row for row in _fts_rows(index) if row[0] == unrelated[0])
    with sqlite3.connect(index / "catalog.sqlite") as conn:
        conn.execute(
            "INSERT INTO chunk(file_path, heading_path, line_start, line_end, content_hash, "
            "mtime, emb_path, text_preview, token_count) "
            "SELECT file_path, heading_path, line_start, line_end, content_hash, "
            "mtime, emb_path, text_preview, token_count FROM chunk WHERE chunk_id=?",
            (original[0],),
        )
        duplicate_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='chunk_fts'"
        ).fetchone():
            conn.execute(
                "INSERT INTO chunk_fts(rowid, file_path, heading_path, text) "
                "SELECT ?, file_path, heading_path, text FROM chunk_fts WHERE rowid=?",
                (duplicate_id, original[0]),
            )

    kb_rag.build_index(config, index_dir=index, force=True)

    rows_after = _rows(index)
    fts_after = _fts_rows(index)
    assert len(rows_after) == 2
    assert next(row for row in rows_after if row[0] == original[0])[0] == original[0]
    unrelated_after = next(row for row in rows_after if row[0] == unrelated[0])
    assert unrelated_after[:5] == unrelated_before[:5]
    assert len(fts_after) == 2
    assert next(row for row in fts_after if row[0] == unrelated[0]) == unrelated_fts_before
    assert duplicate_id not in {row[0] for row in rows_after}
    assert duplicate_id not in {row[0] for row in fts_after}
    assert active_vector.is_file()
    assert unrelated_before[5] == original[5]
    assert kb_catalog_dependency.read_dependency(index / "catalog.sqlite") is not None
