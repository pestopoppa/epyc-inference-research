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


def _all_chunk_rows(index: Path) -> list[tuple]:
    with sqlite3.connect(index / "catalog.sqlite") as conn:
        return conn.execute("SELECT * FROM chunk ORDER BY chunk_id").fetchall()


def _run_writer(
    operation: str, config: kb_rag.CorpusConfig, source: Path, index: Path
) -> dict:
    if operation == "build_index":
        return kb_rag.build_index(config, index_dir=index, force=True)
    if operation == "update_files":
        return kb_rag.update_files([str(source)], config, index_dir=index)
    raise AssertionError(f"unknown fixture writer: {operation}")


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
    config, source = corpus
    index = tmp_path / "index"
    other_source = Path(config.roots[0]) / "two.md"
    other_source.write_text("# Other\n\nunrelated body text\n", encoding="utf-8")
    kb_rag.build_index(config, index_dir=index, force=True)
    source.write_text(
        "## Repeated\nshared body text\n## Repeated\nshared body text\n",
        encoding="utf-8",
    )
    kb_rag.build_index(config, index_dir=index, force=True)
    one_rows = [row for row in _rows(index) if row[1].endswith("one.md")]
    assert len(one_rows) == 2
    original, unrelated = one_rows
    active_vector = index / original[5]
    assert original[2] == unrelated[2]
    assert original[5] == unrelated[5]
    with sqlite3.connect(index / "catalog.sqlite") as conn:
        unrelated_before = conn.execute(
            "SELECT * FROM chunk WHERE chunk_id=?", (unrelated[0],)
        ).fetchone()
        unrelated_vector_before = conn.execute(
            "SELECT emb_path FROM chunk WHERE chunk_id=?", (unrelated[0],)
        ).fetchone()[0]
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

    narrowed_config = kb_rag.CorpusConfig(
        roots=config.roots,
        include_globs=["one.md"],
        exclude_patterns=config.exclude_patterns,
        max_chunk_chars=config.max_chunk_chars,
    )
    kb_rag.build_index(narrowed_config, index_dir=index, force=True)

    rows_after = _rows(index)
    fts_after = _fts_rows(index)
    assert len(rows_after) == 2
    assert next(row for row in rows_after if row[0] == original[0])[0] == original[0]
    with sqlite3.connect(index / "catalog.sqlite") as conn:
        unrelated_after = conn.execute(
            "SELECT * FROM chunk WHERE chunk_id=?", (unrelated[0],)
        ).fetchone()
    assert unrelated_after == unrelated_before
    assert len(fts_after) == 2
    assert next(row for row in fts_after if row[0] == unrelated[0]) == unrelated_fts_before
    assert duplicate_id not in {row[0] for row in rows_after}
    assert duplicate_id not in {row[0] for row in fts_after}
    assert active_vector.is_file()
    assert unrelated_vector_before == original[5]
    assert kb_catalog_dependency.read_dependency(index / "catalog.sqlite") is not None


@pytest.mark.parametrize("operation", ["build_index", "update_files"])
def test_partial_staged_vector_write_preserves_active_state(
    fake_encoder: Mock,
    corpus: tuple[kb_rag.CorpusConfig, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    config, source = corpus
    index = tmp_path / "index"
    kb_rag.build_index(config, index_dir=index, force=True)
    before_rows = _all_chunk_rows(index)
    before_fts = _fts_rows(index)
    before_dependency = kb_catalog_dependency.read_dependency(index / "catalog.sqlite")
    vector = index / _rows(index)[0][5]
    np.savez_compressed(vector, emb=np.full((2, 4), -0.5, dtype=np.float32))
    sentinel_bytes = vector.read_bytes()
    assert before_dependency is not None
    fake_encoder.return_value = np.full((2, 4), 0.75, dtype=np.float32)
    staged_paths: list[Path] = []

    def fail_after_partial_write(staged_file, **arrays) -> None:
        staged_path = Path(staged_file.name)
        staged_paths.append(staged_path)
        assert staged_path.parent == vector.parent
        assert staged_path != vector
        staged_file.write(b"partial synthetic npz")
        raise OSError("fake staged serialization failure")

    monkeypatch.setattr(kb_rag.np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="fake staged serialization failure"):
        _run_writer(operation, config, source, index)

    assert _all_chunk_rows(index) == before_rows
    assert _fts_rows(index) == before_fts
    assert kb_catalog_dependency.read_dependency(index / "catalog.sqlite") == before_dependency
    assert vector.read_bytes() == sentinel_bytes
    assert len(staged_paths) == 1
    assert not staged_paths[0].exists()


def test_partial_staged_write_without_prior_target_publishes_no_vector(
    fake_encoder: Mock,
    corpus: tuple[kb_rag.CorpusConfig, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, source = corpus
    index = tmp_path / "index"

    def fail_after_partial_write(staged_file, **arrays) -> None:
        staged_file.write(b"partial synthetic npz")
        raise OSError("fake staged serialization failure")

    monkeypatch.setattr(kb_rag.np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(OSError, match="fake staged serialization failure"):
        kb_rag.build_index(config, index_dir=index, force=True)

    assert _all_chunk_rows(index) == []
    with sqlite3.connect(index / "catalog.sqlite") as conn:
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='chunk_fts'"
        ).fetchone():
            assert conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0] == 0
    assert kb_catalog_dependency.read_dependency(index / "catalog.sqlite") is None
    assert list((index / "emb").glob("*.npz")) == []
    assert list((index / "emb").glob(".*.tmp.npz")) == []


@pytest.mark.parametrize("operation", ["build_index", "update_files"])
def test_staged_vector_replace_refusal_preserves_active_state(
    fake_encoder: Mock,
    corpus: tuple[kb_rag.CorpusConfig, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    config, source = corpus
    index = tmp_path / "index"
    kb_rag.build_index(config, index_dir=index, force=True)
    before_rows = _all_chunk_rows(index)
    before_fts = _fts_rows(index)
    before_dependency = kb_catalog_dependency.read_dependency(index / "catalog.sqlite")
    vector = index / _rows(index)[0][5]
    np.savez_compressed(vector, emb=np.full((2, 4), -0.5, dtype=np.float32))
    sentinel_bytes = vector.read_bytes()
    assert before_dependency is not None
    fake_encoder.return_value = np.full((2, 4), 0.75, dtype=np.float32)

    def refuse_replace(source_path, target_path) -> None:
        assert Path(target_path) == vector
        raise PermissionError("fake atomic replace refusal")

    monkeypatch.setattr(kb_rag.os, "replace", refuse_replace)
    with pytest.raises(PermissionError, match="fake atomic replace refusal"):
        _run_writer(operation, config, source, index)

    assert _all_chunk_rows(index) == before_rows
    assert _fts_rows(index) == before_fts
    assert kb_catalog_dependency.read_dependency(index / "catalog.sqlite") == before_dependency
    assert vector.read_bytes() == sentinel_bytes
    assert list((index / "emb").glob(".*.tmp.npz")) == []


@pytest.mark.parametrize("operation", ["build_index", "update_files"])
def test_staged_vector_success_keeps_npz_format_and_cleans_temporary(
    fake_encoder: Mock,
    corpus: tuple[kb_rag.CorpusConfig, Path],
    tmp_path: Path,
    operation: str,
) -> None:
    config, source = corpus
    index = tmp_path / "index"

    result = _run_writer(operation, config, source, index)

    rows = _rows(index)
    assert result["chunks_encoded"] == 1
    assert len(rows) == 1
    vector = index / rows[0][5]
    with np.load(vector) as stored:
        np.testing.assert_array_equal(stored["emb"], fake_encoder.return_value)
    assert list((index / "emb").glob(".*.tmp.npz")) == []
    record = kb_catalog_dependency.read_dependency(index / "catalog.sqlite")
    assert record is not None
    assert record["operation"] == operation
