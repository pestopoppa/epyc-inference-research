"""Native catalog dependency records: temporary SQLite and fake encoders only."""
import hashlib
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts.kb_rag import parallel_reembed as parallel
from src.retrieval import colbert_encoder as encoder
from src.retrieval import kb_catalog_dependency as dependency, kb_rag


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / 'catalog.sqlite'
    conn = sqlite3.connect(path)
    conn.executescript(kb_rag._CATALOG_SCHEMA)
    conn.execute("INSERT INTO chunk VALUES (1, 'file.md', '[]', 1, 1, 'hash', 0, 'absent.npz', '', 0)")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def fake(monkeypatch, tmp_path):
    tokenizer = tmp_path / 'fake-tokenizer.json'
    tokenizer.write_bytes(b'exact fake bytes, not a tokenizer')
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    monkeypatch.setattr(encoder, '_TOKENIZER_PATH', tokenizer)
    monkeypatch.setattr(encoder, '_tokenizer_sha256', digest)
    monkeypatch.setattr(encoder, '_tokenizer', object())
    monkeypatch.setattr(encoder, '_session', SimpleNamespace(
        get_outputs=lambda: [SimpleNamespace(shape=[1, 1, 4])]))
    monkeypatch.setattr(encoder, '_MODEL_DIR', tmp_path / 'fake-model')
    monkeypatch.setattr(encoder, '_MODEL_SLOT', 'fake-slot')
    monkeypatch.setattr(encoder, '_generation', 10)
    monkeypatch.setattr(encoder, 'is_available', Mock(return_value=True))
    monkeypatch.setattr(encoder, 'ensure_loaded', Mock(return_value=True))
    monkeypatch.setattr(encoder, 'max_query_tokens', lambda: 32)
    monkeypatch.setattr(encoder, 'max_document_tokens', lambda: 300)
    monkeypatch.setattr(encoder, 'encode', Mock(return_value=np.ones((2, 4))))
    monkeypatch.setattr(encoder, 'prefix_tokens_available', lambda: True)
    monkeypatch.setattr(parallel, '_set_pdeathsig', lambda: pytest.fail('no process controls'))
    monkeypatch.setattr(parallel.mp, 'Pool', lambda *a, **k: pytest.fail('no pool'))
    return digest


def publish(path, operation='remove_files', loaded=None):
    conn = sqlite3.connect(path)
    with dependency.writer_connection(conn):
        return dependency.commit_completed_writer(conn, operation, loaded=loaded)


def test_absent_historical_records_are_unknown_and_read_only(catalog, tmp_path, monkeypatch):
    monkeypatch.setattr(encoder, 'ensure_loaded', lambda: pytest.fail('no load'))
    before = catalog.read_bytes()
    assert dependency.read_dependency(catalog) is None
    assert dependency.export_dependency(catalog) is None
    assert catalog.read_bytes() == before
    absent = tmp_path / 'missing' / 'catalog.sqlite'
    assert dependency.read_dependency(absent) is None
    assert not absent.parent.exists()


def test_native_record_distinguishes_stored_loaded_and_legacy_unknown(catalog):
    loaded = {'tokenizer_sha256': 'a' * 64, 'embedding_dim': 128,
              'model_dir': 'fake', 'model_slot': 'slot', 'generation': 2}
    record = publish(catalog, loaded=loaded)
    assert record['loaded_identity'] == loaded
    assert all(value is None for value in record['stored_identity'].values())
    assert dependency.read_dependency(catalog) == record
    assert json.loads(dependency.export_dependency(catalog)) == record


def test_record_is_atomic_with_final_mutation_and_invisible_until_commit(catalog, monkeypatch):
    conn = sqlite3.connect(catalog)
    observer = sqlite3.connect(catalog)
    original_hash = dependency.logical_digest(conn)
    commit = dependency.commit_completed_writer
    with dependency.writer_connection(conn):
        conn.execute("UPDATE chunk SET text_preview='new'")
        assert dependency.logical_digest(observer) == original_hash
        assert not observer.execute("SELECT 1 FROM sqlite_master WHERE name='catalog_dependency'").fetchone()
        record = commit(conn, 'update_files', loaded=None)
    assert dependency.read_dependency(catalog) == record
    assert observer.execute('SELECT text_preview FROM chunk').fetchone()[0] == 'new'
    observer.close()


@pytest.mark.parametrize('failure', ['hash', 'commit'])
def test_failed_record_or_commit_rolls_back_and_closes(catalog, monkeypatch, failure):
    previous = publish(catalog)
    class FailCommit(sqlite3.Connection):
        def commit(self):
            raise sqlite3.OperationalError('fake commit failure')
    conn = sqlite3.connect(catalog, factory=FailCommit if failure == 'commit' else sqlite3.Connection)
    if failure == 'hash':
        monkeypatch.setattr(dependency, 'logical_digest', Mock(side_effect=ValueError('fake hash failure')))
    with pytest.raises((ValueError, sqlite3.OperationalError)):
        with dependency.writer_connection(conn):
            conn.execute("UPDATE chunk SET text_preview='uncommitted'")
            dependency.commit_completed_writer(conn, 'update_files', loaded=None)
    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute('SELECT 1')
    monkeypatch.undo()
    assert dependency.read_dependency(catalog) == previous


@pytest.mark.parametrize('table', ['chunk', 'index_meta', 'fts'])
def test_content_changes_refuse_old_native_row(catalog, table):
    conn = sqlite3.connect(catalog)
    if table == 'fts':
        conn.execute('CREATE VIRTUAL TABLE chunk_fts USING fts5(file_path, heading_path, text)')
        conn.execute("INSERT INTO chunk_fts VALUES ('f', '[]', 'text')")
        conn.commit()
    conn.close()
    publish(catalog)
    with sqlite3.connect(catalog) as conn:
        if table == 'chunk':
            conn.execute("UPDATE chunk SET text_preview='changed'")
        elif table == 'index_meta':
            conn.execute("INSERT INTO index_meta VALUES ('doc_max_tokens','100')")
        else:
            conn.execute("UPDATE chunk_fts SET text='changed'")
    with pytest.raises(ValueError, match='content no longer matches'):
        dependency.read_dependency(catalog)


@pytest.mark.parametrize('tamper', ['bytes', 'empty', 'schema', 'trigger', 'contract'])
def test_record_and_table_tamper_refused(catalog, tamper):
    publish(catalog)
    with sqlite3.connect(catalog) as conn:
        if tamper == 'bytes':
            conn.execute("UPDATE catalog_dependency SET record_json='{}'")
        elif tamper == 'empty':
            conn.execute('DELETE FROM catalog_dependency')
        elif tamper == 'schema':
            conn.execute('ALTER TABLE catalog_dependency ADD COLUMN extra TEXT')
        elif tamper == 'trigger':
            conn.execute('CREATE TRIGGER evil AFTER UPDATE ON catalog_dependency BEGIN SELECT 1; END')
        else:
            record = json.loads(conn.execute('SELECT record_json FROM catalog_dependency').fetchone()[0])
            record['operation'] = 'historical_backfill'
            body = dependency._bytes(record)
            conn.execute('UPDATE catalog_dependency SET record_json=?, record_sha256=?',
                         (body.decode(), hashlib.sha256(body).hexdigest()))
    with pytest.raises(ValueError):
        dependency.read_dependency(catalog)


def test_own_row_excluded_and_vacuum_preserves_logical_identity(catalog):
    first = publish(catalog)
    second = publish(catalog, operation='update_files')
    assert first['logical_catalog_sha256'] == second['logical_catalog_sha256']
    with sqlite3.connect(catalog) as conn:
        conn.execute('VACUUM')
    assert dependency.read_dependency(catalog) == second
    # Missing vectors were never read or attested.
    assert not (catalog.parent / 'absent.npz').exists()


@pytest.mark.parametrize('operation', ['build_index', 'update_files', 'remove_files',
                                       'parallel_prepare', 'parallel_rebuild'])
def test_every_named_writer_emits_native_record(fake, tmp_path, operation, monkeypatch):
    index = tmp_path / 'index'
    config = kb_rag.CorpusConfig([], [], [])
    if operation == 'build_index':
        assert kb_rag.build_index(config, index_dir=index)['ok']
    elif operation == 'update_files':
        assert kb_rag.update_files([], config, index_dir=index)['ok']
    elif operation == 'remove_files':
        monkeypatch.setattr(encoder, 'ensure_loaded', lambda: pytest.fail('no load for remove'))
        monkeypatch.setattr(encoder, '_session', None)
        monkeypatch.setattr(encoder, '_tokenizer', None)
        assert kb_rag.remove_files([], index_dir=index)['ok']
    elif operation == 'parallel_prepare':
        parallel._prepare_encoding_contract(index)
    else:
        parallel.rebuild_catalog([], index)
    record = dependency.read_dependency(index / 'catalog.sqlite')
    assert record['operation'] == operation
    if operation == 'remove_files':
        assert record['loaded_identity'] is None
        assert all(value is None for value in record['stored_identity'].values())
    else:
        assert record['loaded_identity']['tokenizer_sha256'] == fake
        assert record['loaded_identity']['embedding_dim'] == 4
        assert record['loaded_identity']['generation'] == 10
        assert record['stored_identity']['query_max_tokens'] == '32'
        assert record['stored_identity']['doc_max_tokens'] == '300'


@pytest.mark.parametrize('operation', ['build_index', 'update_files', 'parallel_prepare', 'parallel_rebuild'])
def test_failed_loader_emits_no_native_record(fake, monkeypatch, tmp_path, operation):
    index = tmp_path / 'index'
    monkeypatch.setattr(encoder, 'ensure_loaded', lambda: False)
    config = kb_rag.CorpusConfig([], [], [])
    if operation == 'build_index':
        assert not kb_rag.build_index(config, index_dir=index)['ok']
    elif operation == 'update_files':
        assert not kb_rag.update_files([], config, index_dir=index)['ok']
    else:
        with pytest.raises(RuntimeError, match='failed to load'):
            (parallel._prepare_encoding_contract if operation == 'parallel_prepare'
             else lambda path: parallel.rebuild_catalog([], path))(index)
    assert dependency.read_dependency(index / 'catalog.sqlite') is None


def test_caller_record_failure_rolls_back_deletion_and_closes(fake, catalog, monkeypatch):
    index = catalog.parent
    target = index / 'file.md'
    with sqlite3.connect(catalog) as conn:
        conn.execute('UPDATE chunk SET file_path=?', (str(target.resolve()),))
    original = kb_rag._ensure_catalog
    seen = []
    def connect(*a, **kw):
        conn = original(*a, **kw)
        seen.append(conn)
        return conn
    monkeypatch.setattr(kb_rag, '_ensure_catalog', connect)
    monkeypatch.setattr(dependency, 'logical_digest', Mock(side_effect=ValueError('fake hashing failure')))
    with pytest.raises(ValueError, match='hashing failure'):
        kb_rag.remove_files([str(target)], index_dir=index)
    for conn in seen:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute('SELECT 1')
    with sqlite3.connect(catalog) as conn:
        assert conn.execute('SELECT COUNT(*) FROM chunk').fetchone()[0] == 1
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='catalog_dependency'").fetchone()


@pytest.mark.parametrize('field,value', [('tokenizer_sha256', 'not-a-digest'),
                                          ('embedding_dim', True),
                                          ('generation', -1), ('model_slot', None)])
def test_invalid_loaded_identity_refuses_transaction(catalog, field, value):
    loaded = {'tokenizer_sha256': 'a' * 64, 'embedding_dim': 128,
              'model_dir': 'fake', 'model_slot': 'slot', 'generation': 2}
    loaded[field] = value
    conn = sqlite3.connect(catalog)
    with pytest.raises(ValueError):
        with dependency.writer_connection(conn):
            conn.execute("UPDATE chunk SET text_preview='uncommitted'")
            dependency.commit_completed_writer(conn, 'update_files', loaded=loaded)
    assert dependency.read_dependency(catalog) is None
    with sqlite3.connect(catalog) as check:
        assert check.execute('SELECT text_preview FROM chunk').fetchone()[0] == ''


def test_update_encode_exception_rolls_back_active_deletion_and_closes(fake, catalog, monkeypatch):
    target = catalog.parent / 'file.md'
    target.write_text('# Synthetic\nfixture text\n')
    with sqlite3.connect(catalog) as conn:
        conn.execute('UPDATE chunk SET file_path=?', (str(target.resolve()),))
    monkeypatch.setattr(kb_rag, '_walk_corpus', lambda _: [target.resolve()])
    monkeypatch.setattr(encoder, 'encode', Mock(side_effect=RuntimeError('fake encode exception')))
    with pytest.raises(RuntimeError, match='fake encode exception'):
        kb_rag.update_files([str(target)], kb_rag.CorpusConfig([], [], []), index_dir=catalog.parent)
    assert dependency.read_dependency(catalog) is None
    with sqlite3.connect(catalog) as check:
        assert check.execute('SELECT COUNT(*) FROM chunk').fetchone()[0] == 1


def test_stored_caps_and_identity_are_not_replaced_by_live_values(catalog):
    stored = {'tokenizer_sha256': 'b' * 64, 'embedding_dim': '64',
              'query_max_tokens': '48', 'doc_max_tokens': '256',
              'prefix_convention': 'none'}
    with sqlite3.connect(catalog) as conn:
        conn.executemany('INSERT INTO index_meta VALUES (?, ?)', stored.items())
    loaded = {'tokenizer_sha256': 'a' * 64, 'embedding_dim': 128,
              'model_dir': 'fake', 'model_slot': 'slot', 'generation': 2}
    record = publish(catalog, loaded=loaded)
    assert record['loaded_identity'] == loaded
    assert all(record['stored_identity'][key] == value for key, value in stored.items())
    assert dependency.read_dependency(catalog) == record
    with sqlite3.connect(catalog) as conn:
        assert dict(conn.execute('SELECT key, value FROM index_meta')) == stored
