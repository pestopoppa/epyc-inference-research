"""Index-authoritative caps, synthetic arrays and temporary catalogs only."""

import json
import sqlite3
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.kb_rag import parallel_reembed as parallel
from src.retrieval import colbert_encoder as encoder
from src.retrieval import kb_rag


@pytest.fixture
def fake_encoder(monkeypatch):
    monkeypatch.setattr(encoder, "ensure_loaded", Mock(return_value=True))
    monkeypatch.setattr(encoder, "is_available", lambda: True)
    monkeypatch.setattr(encoder, "max_query_tokens", lambda: 32)
    monkeypatch.setattr(encoder, "max_document_tokens", lambda: 300)
    monkeypatch.setattr(encoder, "_session", None)
    monkeypatch.setattr(encoder, "_tokenizer_sha256", None)
    monkeypatch.setattr(encoder, "encode", Mock(return_value=np.ones((2, 4))))
    monkeypatch.setattr(encoder, "maxsim", Mock(return_value=1.0))
    monkeypatch.setattr(kb_rag.kb_rag_query_telemetry, "record_query_length", Mock())
    monkeypatch.setattr(parallel, "_set_pdeathsig", lambda: None)
    monkeypatch.setattr(parallel, "_WORKER_STATE", {})
    monkeypatch.setenv("COLBERT_ENCODE_ONNX_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")


def catalog(tmp_path, meta=None, populated=False):
    directory = tmp_path / "index"
    directory.mkdir()
    with sqlite3.connect(directory / "catalog.sqlite") as conn:
        conn.executescript(kb_rag._CATALOG_SCHEMA)
        conn.executemany("INSERT INTO index_meta VALUES (?, ?)", list((meta or {}).items()))
        if populated:
            conn.execute(
                "INSERT INTO chunk VALUES (1, 'file.md', '[]', 1, 1, 'old', 0, 'old.npz', '', 0)"
            )
    return directory


@pytest.mark.parametrize("config,expected", [
    ({"query_length": 32, "document_length": 300}, (32, 300)),
    ({}, (48, 256)),
    ({"query_length": True, "document_length": 7}, (48, 256)),
    ({"query_length": "32", "document_length": -1}, (48, 256)),
    (None, (48, 256)),
])
def test_declared_helpers_and_fresh_stamp_without_model(tmp_path, monkeypatch, config, expected):
    model = tmp_path / "model"
    model.mkdir()
    if config is not None:
        (model / "onnx_config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(encoder, "_MODEL_DIR", model)
    monkeypatch.setattr(encoder, "_declared_lengths_cache", {})
    monkeypatch.setattr(encoder, "_session", None)
    monkeypatch.setattr(encoder, "_tokenizer_sha256", None)
    directory = catalog(tmp_path)
    with sqlite3.connect(directory / "catalog.sqlite") as conn:
        kb_rag._writer_convention(conn)
        assert kb_rag._index_token_caps(kb_rag._read_meta(conn)) == expected


@pytest.mark.parametrize("meta,expected", [
    ({}, (48, 256)),
    ({"query_max_tokens": "64"}, (64, 256)),
    ({"doc_max_tokens": "100"}, (48, 100)),
    ({"query_max_tokens": "48", "doc_max_tokens": "256"}, (48, 256)),
])
def test_old_caps_are_authoritative(meta, expected):
    assert kb_rag._index_token_caps(meta) == expected


@pytest.mark.parametrize("key", ["query_max_tokens", "doc_max_tokens"])
@pytest.mark.parametrize("value", ["", "bad", "0", "-1", "1.5", " 48", "٣٢"])
def test_invalid_cap_parser(key, value):
    with pytest.raises(RuntimeError, match="invalid stored"):
        kb_rag._index_token_caps({key: value})


@pytest.mark.parametrize("operation", ["build", "update", "parallel", "rebuild", "main"])
def test_bad_stored_caps_refuse_before_load_or_catalog_mutation(
    fake_encoder, tmp_path, monkeypatch, operation,
):
    directory = catalog(tmp_path, {"doc_max_tokens": "bad"}, populated=True)
    before = (directory / "catalog.sqlite").read_bytes()
    monkeypatch.setattr(parallel.mp, "Pool", Mock(side_effect=AssertionError("no pool")))
    config = kb_rag.CorpusConfig([], [], [])
    with pytest.raises(RuntimeError, match="invalid stored"):
        if operation == "build":
            kb_rag.build_index(config, index_dir=directory, force=True)
        elif operation == "update":
            kb_rag.update_files(["file.md"], config, index_dir=directory)
        elif operation == "parallel":
            parallel.encode_missing([("new.npz", "text", encoder.ROLE_NONE)], directory, 1, 1, "fake")
        elif operation == "rebuild":
            parallel.rebuild_catalog([], directory)
        else:
            monkeypatch.setattr(sys, "argv", ["parallel", "run", "--index-dir", str(directory)])
            monkeypatch.setattr(
                parallel, "_load_or_build_manifest",
                Mock(side_effect=AssertionError("no manifest or chunking")),
            )
            parallel.main()
    assert (directory / "catalog.sqlite").read_bytes() == before
    assert not (directory / "emb").exists()
    encoder.ensure_loaded.assert_not_called()
    encoder.encode.assert_not_called()
    parallel.mp.Pool.assert_not_called()


def test_partial_metadata_empty_catalog_is_not_restamped(fake_encoder, tmp_path):
    meta = {"doc_max_tokens": "128", "stamped_at": "historical"}
    directory = catalog(tmp_path, meta)
    with sqlite3.connect(directory / "catalog.sqlite") as conn:
        assert kb_rag._writer_convention(conn) == encoder.LEGACY_CONVENTION
        assert kb_rag._read_meta(conn) == meta
    assert parallel._prepare_encoding_contract(directory) == (128, encoder.ROLE_NONE)


def test_bad_query_cap_refuses_before_encode(fake_encoder, tmp_path):
    directory = catalog(tmp_path, {"query_max_tokens": "bad"}, populated=True)
    with pytest.raises(RuntimeError, match="invalid stored"):
        kb_rag.query("query", index_dir=directory)
    encoder.encode.assert_not_called()
    encoder.maxsim.assert_not_called()


@pytest.mark.parametrize("legacy", [False, True])
def test_build_update_query_and_telemetry_keep_existing_caps(
    fake_encoder, tmp_path, monkeypatch, legacy,
):
    source = tmp_path / "file.md"
    source.write_text("# Fixture\ncontent", encoding="utf-8")
    meta = {} if legacy else {
        "prefix_convention": encoder.PREFIX_CONVENTION,
        "query_max_tokens": "48", "doc_max_tokens": "256", "stamped_at": "original",
    }
    directory = catalog(tmp_path, meta, populated=True)
    monkeypatch.setattr(kb_rag, "_walk_corpus", lambda _: [source])
    config = kb_rag.CorpusConfig([], [], [])
    kb_rag.build_index(config, index_dir=directory, force=True)
    assert encoder.encode.call_args.args[1] == 256
    kb_rag.update_files([str(source)], config, index_dir=directory)
    assert encoder.encode.call_args.args[1] == 256
    kb_rag.query("query", index_dir=directory)
    assert encoder.encode.call_args.args[1] == 48
    assert encoder.encode.call_args.kwargs["role"] == (
        encoder.ROLE_NONE if legacy else encoder.ROLE_QUERY
    )
    assert kb_rag.kb_rag_query_telemetry.record_query_length.call_args.kwargs["cap"] == 48
    with sqlite3.connect(directory / "catalog.sqlite") as conn:
        assert kb_rag._read_meta(conn) == meta


def test_fresh_writer_and_query_use_declared_caps(fake_encoder, tmp_path, monkeypatch):
    source = tmp_path / "file.md"
    source.write_text("# Fixture\ncontent", encoding="utf-8")
    monkeypatch.setattr(kb_rag, "_walk_corpus", lambda _: [source])
    directory = tmp_path / "fresh"
    kb_rag.build_index(kb_rag.CorpusConfig([], [], []), index_dir=directory)
    assert encoder.encode.call_args.args[1] == 300
    kb_rag.query("query", index_dir=directory)
    assert encoder.encode.call_args.args[1] == 32
    assert kb_rag.kb_rag_query_telemetry.record_query_length.call_args.kwargs["cap"] == 32


@pytest.mark.parametrize("legacy", [False, True])
def test_parallel_worker_and_rebuild_preserve_caps_and_convention(fake_encoder, tmp_path, legacy):
    meta = {} if legacy else {
        "prefix_convention": encoder.PREFIX_CONVENTION,
        "query_max_tokens": "48", "doc_max_tokens": "256", "stamped_at": "original",
    }
    directory = catalog(tmp_path, meta, populated=True)
    parallel._encode_init(str(directory), 1, [])
    assert parallel._WORKER_STATE["max_tokens"] == 256
    assert parallel._WORKER_STATE["role"] == (
        encoder.ROLE_NONE if legacy else encoder.ROLE_DOCUMENT
    )
    parallel.rebuild_catalog([], directory)
    with sqlite3.connect(directory / "catalog.sqlite") as conn:
        assert kb_rag._read_meta(conn) == meta


def test_parallel_fresh_contract_stamps_before_worker(fake_encoder, tmp_path):
    directory = tmp_path / "fresh"
    assert parallel._prepare_encoding_contract(directory) == (300, encoder.ROLE_DOCUMENT)
    parallel._encode_init(str(directory), 1, [])
    assert parallel._WORKER_STATE["max_tokens"] == 300
    with sqlite3.connect(directory / "catalog.sqlite") as conn:
        assert kb_rag._index_token_caps(kb_rag._read_meta(conn)) == (32, 300)


def test_parallel_rejects_role_mismatch_before_pool(fake_encoder, tmp_path, monkeypatch):
    directory = catalog(tmp_path, populated=True)
    monkeypatch.setattr(parallel.mp, "Pool", Mock(side_effect=AssertionError("no pool")))
    with pytest.raises(RuntimeError, match="stored index convention"):
        parallel.encode_missing([("new.npz", "text", encoder.ROLE_DOCUMENT)], directory, 1, 1, "fake")
    parallel.mp.Pool.assert_not_called()
    encoder.encode.assert_not_called()


def test_parallel_refuses_orphan_embeddings_new_identity(fake_encoder, tmp_path):
    directory = tmp_path / "index"
    (directory / "emb").mkdir(parents=True)
    (directory / "emb" / "old.npz").write_bytes(b"orphan fixture")
    with pytest.raises(RuntimeError, match="without catalog identity"):
        parallel._prepare_encoding_contract(directory)
    with sqlite3.connect(directory / "catalog.sqlite") as conn:
        assert kb_rag._read_meta(conn) == {}


def test_parallel_tokenizer_mismatch_refuses_before_rebuild(fake_encoder, tmp_path):
    directory = catalog(tmp_path, {
        "prefix_convention": encoder.PREFIX_CONVENTION,
        "tokenizer_sha256": "known other identity",
    }, populated=True)
    before = (directory / "catalog.sqlite").read_bytes()
    with pytest.raises(RuntimeError, match="tokenizer"):
        parallel.rebuild_catalog([], directory)
    assert (directory / "catalog.sqlite").read_bytes() == before
    encoder.encode.assert_not_called()
