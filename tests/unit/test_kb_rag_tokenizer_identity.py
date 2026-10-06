"""Tokenizer provenance guards: temporary bytes/SQLite and fake encoders only."""

import hashlib
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.retrieval import colbert_encoder as encoder
from src.retrieval import kb_rag


@pytest.fixture
def identity(tmp_path, monkeypatch):
    path = tmp_path / "tokenizer.json"
    path.write_bytes(b'{"fixture": "original"}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(encoder, "_TOKENIZER_PATH", path)
    monkeypatch.setattr(encoder, "_tokenizer_sha256", digest)
    monkeypatch.setattr(
        encoder, "_session",
        SimpleNamespace(get_outputs=lambda: [SimpleNamespace(shape=[1, 1, 128])]),
    )
    monkeypatch.setattr(encoder, "_tokenizer", object())
    monkeypatch.setattr(encoder, "ensure_loaded", lambda: True)
    monkeypatch.setattr(encoder, "is_available", lambda: True)
    monkeypatch.setattr(encoder, "encode", Mock(side_effect=AssertionError("no encode")))
    monkeypatch.setattr(encoder, "maxsim", Mock(side_effect=AssertionError("no scoring")))
    return path, digest


def catalog(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    conn = sqlite3.connect(index_dir / "catalog.sqlite")
    conn.executescript(kb_rag._CATALOG_SCHEMA)
    return index_dir, conn


def test_stamp_and_matching_identity(identity, tmp_path):
    _, digest = identity
    _, conn = catalog(tmp_path)
    kb_rag._stamp_meta(conn, encoder.PREFIX_CONVENTION)
    meta = kb_rag._read_meta(conn)
    assert meta["tokenizer_sha256"] == digest
    kb_rag._check_tokenizer_identity(meta)
    conn.close()


@pytest.mark.parametrize("failure", ["changed", "missing", "unknown_loaded", "other_loaded"])
def test_known_stamp_refuses_unverified_identity(identity, monkeypatch, failure):
    path, digest = identity
    if failure == "changed":
        path.write_bytes(b"different bytes at the same path")
    elif failure == "missing":
        path.unlink()
    else:
        monkeypatch.setattr(
            encoder, "_tokenizer_sha256",
            None if failure == "unknown_loaded" else "0" * 64,
        )
    with pytest.raises(RuntimeError, match="tokenizer"):
        kb_rag._check_tokenizer_identity({"tokenizer_sha256": digest})


def test_legacy_unknown_identity_does_not_read_or_stamp(identity, tmp_path, monkeypatch):
    path, _ = identity
    path.unlink()
    monkeypatch.setattr(encoder, "_tokenizer_sha256", None)
    _, conn = catalog(tmp_path)
    conn.execute(
        "INSERT INTO chunk VALUES (1, 'file.md', '[]', 1, 1, 'hash', 0, 'old.npz', '', 0)"
    )
    assert kb_rag._writer_convention(conn) == encoder.LEGACY_CONVENTION
    assert kb_rag._read_meta(conn) == {}
    kb_rag._check_tokenizer_identity({})
    conn.close()


def test_existing_writer_preserves_stamp(identity, tmp_path):
    _, conn = catalog(tmp_path)
    kb_rag._stamp_meta(conn, encoder.PREFIX_CONVENTION)
    before = kb_rag._read_meta(conn)
    assert kb_rag._writer_convention(conn) == encoder.PREFIX_CONVENTION
    assert kb_rag._read_meta(conn) == before
    conn.close()


def test_remove_stamped_files_needs_no_loaded_encoder(identity, tmp_path, monkeypatch):
    _, _ = identity
    index_dir, conn = catalog(tmp_path)
    source = tmp_path / "removed.md"
    kb_rag._stamp_meta(conn, encoder.PREFIX_CONVENTION)
    conn.execute(
        "INSERT INTO chunk VALUES (1, ?, '[]', 1, 1, 'hash', 0, 'old.npz', '', 0)",
        (str(source),),
    )
    before = kb_rag._read_meta(conn)
    conn.commit()
    conn.close()
    monkeypatch.setattr(encoder, "_session", None)
    monkeypatch.setattr(encoder, "_tokenizer", None)
    monkeypatch.setattr(encoder, "_tokenizer_sha256", None)
    monkeypatch.setattr(
        encoder, "ensure_loaded", Mock(side_effect=AssertionError("no model load")),
    )
    result = kb_rag.remove_files([str(source)], index_dir=index_dir)
    assert result["ok"]
    with sqlite3.connect(index_dir / "catalog.sqlite") as check:
        assert check.execute("SELECT COUNT(*) FROM chunk").fetchone()[0] == 0
        assert kb_rag._read_meta(check) == before
    encoder.ensure_loaded.assert_not_called()
    encoder.encode.assert_not_called()


@pytest.mark.parametrize("operation", ["query", "force_build", "update"])
def test_mismatch_refuses_before_encode_or_writer_delete(
    identity, tmp_path, monkeypatch, operation,
):
    path, _ = identity
    index_dir, conn = catalog(tmp_path)
    kb_rag._stamp_meta(conn, encoder.PREFIX_CONVENTION)
    source = tmp_path / "file.md"
    source.write_text("# Fixture\ncontent", encoding="utf-8")
    conn.execute(
        "INSERT INTO chunk VALUES (1, ?, '[]', 1, 1, 'hash', 0, 'old.npz', '', 0)",
        (str(source),),
    )
    before = kb_rag._read_meta(conn)
    conn.commit()
    conn.close()
    path.write_bytes(b"mutated")
    config = kb_rag.CorpusConfig([], [], [])
    monkeypatch.setattr(kb_rag, "_walk_corpus", lambda _: [source])
    with pytest.raises(RuntimeError, match="tokenizer"):
        if operation == "query":
            kb_rag.query("fixture", index_dir=index_dir)
        elif operation == "force_build":
            kb_rag.build_index(config, index_dir=index_dir, force=True)
        else:
            kb_rag.update_files([str(source)], config, index_dir=index_dir)
    with sqlite3.connect(index_dir / "catalog.sqlite") as check:
        assert check.execute("SELECT COUNT(*) FROM chunk").fetchone()[0] == 1
        assert kb_rag._read_meta(check) == before
    encoder.encode.assert_not_called()
    encoder.maxsim.assert_not_called()


def test_new_stamp_refuses_cached_file_disagreement(identity, tmp_path):
    path, _ = identity
    _, conn = catalog(tmp_path)
    path.write_bytes(b"new bytes")
    with pytest.raises(RuntimeError, match="tokenizer"):
        kb_rag._stamp_meta(conn, encoder.PREFIX_CONVENTION)
    assert kb_rag._read_meta(conn) == {}
    conn.close()


def test_loader_hashes_exact_parsed_buffer(tmp_path, monkeypatch):
    path = tmp_path / "tokenizer.json"
    original = b'{"fixture": "loaded"}'
    path.write_bytes(original)
    fake_tokenizer = SimpleNamespace(token_to_id=lambda _: 1)

    def parse(text):
        assert text == original.decode("utf-8")
        path.write_bytes(b"changed during parse")
        return fake_tokenizer

    fake_session = SimpleNamespace(get_inputs=lambda: [])
    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(
            SessionOptions=SimpleNamespace,
            InferenceSession=lambda *args, **kwargs: fake_session,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(
            Tokenizer=SimpleNamespace(from_str=parse),
        ),
    )
    for name, value in {
        "_session": None, "_tokenizer": None, "_tokenizer_sha256": None,
        "_TOKENIZER_PATH": path, "_count_tokenizer": None, "_input_names": (),
        "_do_lower_case": False, "_prefix_tokens_ok": False,
        "_query_prefix": "[Q] ", "_document_prefix": "[D] ",
    }.items():
        monkeypatch.setattr(encoder, name, value)
    monkeypatch.setattr(encoder, "is_available", lambda: True)
    monkeypatch.setattr(encoder, "_prefix_encodes_to_one_token", lambda *args: True)
    monkeypatch.setattr(
        encoder, "_load_declared_prefixes", lambda _: ("[Q] ", "[D] "),
    )
    monkeypatch.setattr(encoder, "_load_declared_prefix_ids", lambda _: {})
    monkeypatch.setattr(encoder, "_load_declared_config", lambda _: {})
    assert encoder.ensure_loaded()
    assert encoder._tokenizer_sha256 == hashlib.sha256(original).hexdigest()
    with pytest.raises(RuntimeError, match="tokenizer"):
        kb_rag._check_tokenizer_identity({"tokenizer_sha256": encoder._tokenizer_sha256})


def test_refresh_clears_loaded_identity(identity, tmp_path, monkeypatch):
    monkeypatch.setattr(
        encoder, "resolve_model_dir", lambda: (tmp_path / "other", "other"),
    )
    for name in (
        "_MODEL_DIR", "_MODEL_SLOT", "_MODEL_PATH", "_count_tokenizer",
        "_prefix_tokens_ok", "_input_names", "_do_lower_case",
    ):
        monkeypatch.setattr(encoder, name, getattr(encoder, name))
    encoder.refresh_model_dir()
    assert encoder._tokenizer_sha256 is None
    assert encoder._tokenizer is None


def test_loader_failure_clears_partial_session_and_identity(tmp_path, monkeypatch):
    path = tmp_path / "tokenizer.json"
    path.write_bytes(b"invalid tokenizer fixture")
    created = []

    def session(*args, **kwargs):
        loaded = object()
        created.append(loaded)
        return loaded

    def parse(text):
        assert text == "invalid tokenizer fixture"
        assert encoder._session is created[0]
        raise ValueError("invalid tokenizer")

    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(
            SessionOptions=SimpleNamespace, InferenceSession=session,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(
            Tokenizer=SimpleNamespace(from_str=parse),
        ),
    )
    monkeypatch.setattr(encoder, "_session", None)
    monkeypatch.setattr(encoder, "_tokenizer", None)
    monkeypatch.setattr(encoder, "_tokenizer_sha256", "previous identity")
    monkeypatch.setattr(encoder, "_count_tokenizer", None)
    monkeypatch.setattr(encoder, "_TOKENIZER_PATH", path)
    monkeypatch.setattr(encoder, "is_available", lambda: True)
    assert not encoder.ensure_loaded()
    assert len(created) == 1
    assert encoder._session is None
    assert encoder._tokenizer is None
    assert encoder._tokenizer_sha256 is None
