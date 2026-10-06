"""Per-process encoder transactions using fake sessions/tokenizers only."""
import hashlib
import importlib.util
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.retrieval import colbert_encoder as encoder
from src.retrieval import kb_rag, kb_rag_query_telemetry as telemetry
# Load this thin module directly: package initialization registers unrelated
# tools and is outside the hermetic encoder contract.
_reranker_path = Path(__file__).resolve().parents[2] / 'src/tools/web/colbert_reranker.py'
_reranker_spec = importlib.util.spec_from_file_location('s9_reranker_fake', _reranker_path)
reranker = importlib.util.module_from_spec(_reranker_spec)
_reranker_spec.loader.exec_module(reranker)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    class Tokenizer:
        def __init__(self):
            self.length = 1
        def enable_truncation(self, *, max_length):
            self.length = max_length
        def enable_padding(self, *, length):
            assert self.length == length
        def no_padding(self):
            pass
        def no_truncation(self):
            pass
        def token_to_id(self, text):
            return 1
        def to_str(self):
            return '{}'
        def encode(self, text, **kwargs):
            n = self.length if not kwargs else 1
            return SimpleNamespace(ids=[1] * n, attention_mask=[1] * n)
    tok = Tokenizer()
    session = SimpleNamespace(
        run=lambda _, feed: [np.ones((1, feed['input_ids'].shape[1], 2))],
        get_inputs=lambda: [SimpleNamespace(name='input_ids')],
        get_outputs=lambda: [SimpleNamespace(shape=[1, 1, 2])],
    )
    for name, value in {'_session': session, '_tokenizer': tok,
                        '_count_tokenizer': None, '_tokenizer_sha256': None,
                        '_query_prefix': encoder._FALLBACK_QUERY_PREFIX,
                        '_document_prefix': encoder._FALLBACK_DOCUMENT_PREFIX,
                        '_MODEL_SLOT': 'fake', '_input_names': ('input_ids',),
                        '_prefix_tokens_ok': True, '_do_lower_case': False,
                        '_generation': 10, '_declared_lengths_cache': {}}.items():
        monkeypatch.setattr(encoder, name, value)
    monkeypatch.setenv('COLBERT_ENCODE_ONNX_THREADS', '1')
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setattr(encoder, '_MODEL_DIR', tmp_path)
    monkeypatch.setattr(encoder, '_MODEL_PATH', tmp_path / 'model_int8.onnx')
    monkeypatch.setattr(encoder, '_TOKENIZER_PATH', tmp_path / 'tokenizer.json')
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(
        Tokenizer=SimpleNamespace(from_str=lambda _: Tokenizer())))
    return tok, session


def contend(operation, refresh):
    """Events prove a competing refresh cannot acquire during the operation."""
    entered, attempted, released = (threading.Event() for _ in range(3))
    errors = []
    original_lock = encoder._state_lock
    contender_id = None
    class ObservedLock:
        def __enter__(self):
            if threading.get_ident() == contender_id and not attempted.is_set():
                # Prove this thread attempted acquisition while the operation
                # owns the lock, rather than relying on scheduler timing.
                acquired = original_lock.acquire(blocking=False)
                if acquired:
                    original_lock.release()
                    raise AssertionError('competing refresh acquired during transaction')
                attempted.set()
            original_lock.acquire()
            return self
        def __exit__(self, *exc):
            original_lock.release()
    encoder._state_lock = ObservedLock()
    def competitor():
        nonlocal contender_id
        contender_id = threading.get_ident()
        try:
            assert entered.wait(5)
            refresh()
            released.set()
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=competitor)
    thread.start()
    try:
        operation(entered, attempted, released)
    finally:
        entered.set()
        thread.join(5)
        encoder._state_lock = original_lock
    assert not thread.is_alive()
    assert not errors
    assert released.is_set()


def test_reentrant_metadata_does_not_load(fake, monkeypatch):
    monkeypatch.setattr(encoder, 'ensure_loaded', lambda: pytest.fail('unexpected load'))
    with encoder.locked_state() as generation:
        assert encoder.generation() == generation
        assert encoder.max_query_tokens() == 48
        assert encoder.max_document_tokens() == 256
        assert encoder.prefix_for_role(encoder.ROLE_NONE) == ''


def test_encode_serializes_mutable_padding_and_refresh(fake, monkeypatch, tmp_path):
    tok, session = fake
    observations = []
    def operation(entered, attempted, released):
        def run(_, feed):
            entered.set()
            assert attempted.wait(5)
            assert not released.is_set()
            assert encoder.generation() == 10  # reentrant callback
            observations.append((tok.length, feed['input_ids'].shape[1]))
            return [np.ones((1, tok.length, 2))]
        session.run = run
        assert encoder.encode('text', 7, role=encoder.ROLE_NONE).shape == (7, 2)
    monkeypatch.setattr(encoder, 'resolve_model_dir', lambda: (tmp_path / 'next', 'next'))
    contend(operation, encoder.refresh_model_dir)
    assert observations == [(7, 7)]
    assert encoder.generation() == 11
    assert encoder._session is encoder._tokenizer is encoder._tokenizer_sha256 is None


def test_count_serializes_copy_and_refresh(fake, monkeypatch, tmp_path):
    class Counter:
        def no_padding(self): pass
        def no_truncation(self): pass
        def encode(self, text):
            entered.set()
            assert attempted.wait(5)
            assert not released.is_set()
            return SimpleNamespace(ids=[1, 2, 3])
    monkeypatch.setattr(encoder, '_count_tokenizer', Counter())
    monkeypatch.setattr(encoder, 'resolve_model_dir', lambda: (tmp_path / 'next', 'next'))
    def operation(e, a, r):
        nonlocal entered, attempted, released
        entered, attempted, released = e, a, r
        assert encoder.count_tokens('text', role=encoder.ROLE_NONE) == 3
    entered = attempted = released = None
    contend(operation, encoder.refresh_model_dir)
    assert encoder._count_tokenizer is None


@pytest.mark.parametrize('fails', [False, True])
def test_loader_publishes_only_complete_state_exact_buffer(fake, monkeypatch, fails):
    tok, session = fake
    raw = b'{"fake":"exact bytes"}'
    encoder._MODEL_PATH.write_bytes(b'not a model')
    encoder._TOKENIZER_PATH.write_bytes(raw)
    monkeypatch.setattr(encoder, '_session', None)
    monkeypatch.setattr(encoder, '_tokenizer', None)
    def construct(text):
        assert encoder._session is encoder._tokenizer is encoder._tokenizer_sha256 is None
        assert text.encode() == raw
        if fails:
            raise ValueError('fake parse failure')
        return tok
    monkeypatch.setitem(sys.modules, 'onnxruntime', SimpleNamespace(
        SessionOptions=SimpleNamespace, InferenceSession=lambda *a, **k: session))
    monkeypatch.setitem(sys.modules, 'tokenizers', SimpleNamespace(
        Tokenizer=SimpleNamespace(from_str=construct)))
    assert encoder.ensure_loaded() is (not fails)
    if fails:
        assert encoder._session is encoder._tokenizer is encoder._count_tokenizer is None
        assert encoder._tokenizer_sha256 is None
        assert encoder._input_names == ()
        assert not encoder._prefix_tokens_ok and not encoder._do_lower_case
        assert encoder.generation() == 10
    else:
        assert encoder._session is session and encoder._tokenizer is tok
        assert encoder._tokenizer_sha256 == hashlib.sha256(raw).hexdigest()
        assert encoder.generation() == 11


def test_declared_cap_cache_is_serialized(fake, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(encoder, '_load_declared_config', lambda _: calls.append(1) or
                        {'query_length': 32, 'document_length': 300})
    with encoder.locked_state():
        assert encoder.max_query_tokens() == 32
        assert encoder.max_document_tokens() == 300
    assert calls == [1]


def test_kb_validation_and_encode_hold_same_generation(fake, monkeypatch, tmp_path):
    index = tmp_path / 'index'
    conn = kb_rag._ensure_catalog(index)
    conn.close()
    observed = []
    monkeypatch.setattr(encoder, 'ensure_loaded', lambda: True)
    monkeypatch.setattr(encoder, 'resolve_model_dir', lambda: (tmp_path / 'next', 'next'))
    def operation(entered, attempted, released):
        def validate(meta):
            entered.set()
            assert attempted.wait(5)
            assert not released.is_set()
            observed.append(('validate', encoder.generation()))
        def encode(*a, **k):
            observed.append(('encode', encoder.generation()))
            assert not released.is_set()
            return None
        monkeypatch.setattr(kb_rag, '_warn_on_encoder_drift', validate)
        monkeypatch.setattr(encoder, 'encode', encode)
        assert kb_rag.query('text', index_dir=index) == []
    contend(operation, encoder.refresh_model_dir)
    assert observed == [('validate', 10), ('encode', 10)]


def test_reranker_query_and_docs_hold_same_generation(fake, monkeypatch, tmp_path):
    observed = []
    monkeypatch.setattr(encoder, 'ensure_loaded', lambda: True)
    monkeypatch.setattr(encoder, 'resolve_model_dir', lambda: (tmp_path / 'next', 'next'))
    def operation(entered, attempted, released):
        def encode(*a, **k):
            observed.append(encoder.generation())
            if len(observed) == 1:
                entered.set()
                assert attempted.wait(5)
            assert not released.is_set()
            return np.ones((1, 2))
        monkeypatch.setattr(encoder, 'encode', encode)
        assert len(reranker.rerank_snippets('query', [{'snippet': 'one'}, {'snippet': 'two'}])) == 2
    contend(operation, encoder.refresh_model_dir)
    assert observed == [10, 10, 10]


def test_telemetry_count_and_metadata_hold_same_generation(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(telemetry, 'log_path', lambda: tmp_path / 'log.jsonl')
    monkeypatch.setattr(encoder, 'resolve_model_dir', lambda: (tmp_path / 'next', 'next'))
    def operation(entered, attempted, released):
        def count(*a, **k):
            entered.set()
            assert attempted.wait(5)
            assert not released.is_set()
            return 7
        monkeypatch.setattr(encoder, 'count_tokens', count)
        record = telemetry.record_query_length('text', cap=48, role=encoder.ROLE_NONE,
            prefix_convention='none', index_dir=tmp_path)
        assert record['encoder_model_dir'] == str(tmp_path)
    contend(operation, encoder.refresh_model_dir)


@pytest.fixture
def parallel():
    path = Path(__file__).resolve().parents[2] / 'scripts/kb_rag/parallel_reembed.py'
    spec = importlib.util.spec_from_file_location('s9_parallel_fake', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worker_contract_rejects_stale_generation_before_encode(fake, monkeypatch, tmp_path, parallel):
    monkeypatch.setattr(parallel, '_set_pdeathsig', lambda: None)
    monkeypatch.setattr(parallel, '_parallel_contract', lambda *a, **k: (256, encoder.ROLE_NONE))
    parallel._encode_init(str(tmp_path), 1, [])
    assert parallel._WORKER_STATE['generation'] == 10
    monkeypatch.setattr(encoder, '_generation', 11)
    monkeypatch.setattr(encoder, 'encode', lambda *a, **k: pytest.fail('stale encode'))
    with pytest.raises(RuntimeError, match='generation changed'):
        parallel._encode_task(('emb/new.npz', 'text', encoder.ROLE_NONE))
    assert not (tmp_path / 'emb').exists()


def test_pool_construction_occurs_outside_encoder_lock(fake, monkeypatch, tmp_path, parallel):
    monkeypatch.setattr(parallel, '_prepare_encoding_contract',
                        encoder.state_transaction(lambda _: (256, encoder.ROLE_NONE)))
    class PoolReached(Exception): pass
    def pool(**kwargs):
        # A different thread must acquire, proving no inherited/held parent lock.
        acquired = threading.Event()
        def probe():
            with encoder.locked_state():
                acquired.set()
        thread = threading.Thread(target=probe)
        thread.start()
        thread.join(5)
        assert acquired.is_set() and not thread.is_alive()
        raise PoolReached()
    monkeypatch.setattr(parallel.mp, 'Pool', pool)
    with pytest.raises(PoolReached):
        parallel.encode_missing([('emb.npz', 'text', encoder.ROLE_NONE)], tmp_path, 1, 1, 'fake')


def test_two_encodes_cannot_interleave_tokenizer_settings(fake):
    tok, session = fake
    shapes = []
    def operation(entered, attempted, released):
        def run(_, feed):
            n = feed['input_ids'].shape[1]
            if n == 7:
                entered.set()
                assert attempted.wait(5)
                assert not released.is_set()
                assert tok.length == 7
            return [np.ones((1, n, 2))]
        session.run = run
        shapes.append(encoder.encode('first', 7, role=encoder.ROLE_NONE).shape)
    contend(operation, lambda: shapes.append(
        encoder.encode('other', 3, role=encoder.ROLE_NONE).shape))
    assert sorted(shapes) == [(3, 2), (7, 2)]


def test_worker_initializer_captures_contract_under_lock(fake, monkeypatch, tmp_path, parallel):
    monkeypatch.setattr(parallel, '_set_pdeathsig', lambda: None)
    monkeypatch.setattr(encoder, 'resolve_model_dir', lambda: (tmp_path / 'next', 'next'))
    def operation(entered, attempted, released):
        def contract(*args, **kwargs):
            entered.set()
            assert attempted.wait(5)
            assert not released.is_set()
            return 300, encoder.ROLE_DOCUMENT
        monkeypatch.setattr(parallel, '_parallel_contract', contract)
        parallel._encode_init(str(tmp_path), 1, [])
        assert parallel._WORKER_STATE['max_tokens'] == 300
        assert parallel._WORKER_STATE['role'] == encoder.ROLE_DOCUMENT
        assert parallel._WORKER_STATE['generation'] == 10
    contend(operation, encoder.refresh_model_dir)


def test_worker_output_publication_holds_same_generation(fake, monkeypatch, tmp_path, parallel):
    monkeypatch.setattr(parallel, '_set_pdeathsig', lambda: None)
    monkeypatch.setattr(parallel, '_parallel_contract', lambda *a, **k: (256, encoder.ROLE_NONE))
    parallel._encode_init(str(tmp_path), 1, [])
    monkeypatch.setattr(encoder, 'resolve_model_dir', lambda: (tmp_path / 'next', 'next'))
    replace = parallel.os.replace
    def operation(entered, attempted, released):
        def publish(source, destination):
            entered.set()
            assert attempted.wait(5)
            assert not released.is_set()
            assert encoder.generation() == 10
            replace(source, destination)
        monkeypatch.setattr(parallel.os, 'replace', publish)
        assert parallel._encode_task(('emb/new.npz', 'text', encoder.ROLE_NONE)) == (
            'emb/new.npz', 256, 'encoded')
        assert (tmp_path / 'emb/new.npz').is_file()
    contend(operation, encoder.refresh_model_dir)
