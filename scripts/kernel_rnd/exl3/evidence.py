"""VB-EXL3-CPU-GFX90A prospective writers and strict projection grammar.

Rows are one directed metric or one exact decided proposition. Attestation and
all verifier read-set bytes are reopened at projection. No grading rule lives
here: root's adapter constructs ClaimTuple and the shared grader decides.
"""
from __future__ import annotations
import datetime
import hashlib
import json
import math
from pathlib import Path

PRODUCER_ID = 'epyc.exl3.evidence/v1'
AUTHORITY = 'experimental_no_promotion'
MEASUREMENT = 'epyc.exl3.measurement.v1'
VERIFIER = 'epyc.exl3.verifier.v1'
IDENTITIES = {'model', 'artifact', 'source', 'binary', 'library', 'toolchain', 'hardware', 'residency'}
COMMON = set('schema producer_id producer_sha256 run_id row_id self_sha256 locator date category protocol_id protocol_eligible attestation_path attestation_sha256 arm comparator authority identities claim'.split())
MEASURE_FIELDS = set('backend operator shape metric value unit metric_direction repetitions reps_basis raw_vector aggregation'.split())
VERIFY_FIELDS = set('fixture backend path decided_proposition verdict checker fixture_sha256 read_set read_set_sha256'.split())


class EvidenceRefusal(ValueError):
    pass


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def producer_hash():
    return file_hash(__file__)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _exact(obj, keys):
    if not isinstance(obj, dict) or set(obj) != keys:
        raise EvidenceRefusal('unknown or missing evidence fields')


def _json(path):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise EvidenceRefusal('duplicate evidence key')
            d[k] = v
        return d
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(EvidenceRefusal('nonfinite JSON')))


def _file(desc):
    _exact(desc, {'path', 'sha256'})
    if not _text(desc['path']) or not _hash(desc['sha256']) or not Path(desc['path']).is_file() or file_hash(desc['path']) != desc['sha256']:
        raise EvidenceRefusal('missing or drifted evidence file')


def locator(row):
    if row['schema'] == MEASUREMENT:
        return [row['run_id'], row['arm'], row['backend'], row['operator'], row['shape'], row['metric']]
    return [row['run_id'], row['fixture'], row['backend'], row['path'], row['decided_proposition']]


def validate(row, *, reopen=True):
    schema = row.get('schema') if isinstance(row, dict) else None
    if schema not in {MEASUREMENT, VERIFIER}:
        raise EvidenceRefusal('unknown or pre-hook evidence schema')
    _exact(row, COMMON | (MEASURE_FIELDS if schema == MEASUREMENT else VERIFY_FIELDS))
    if row['producer_id'] != PRODUCER_ID or row['producer_sha256'] != producer_hash():
        raise EvidenceRefusal('producer identity/hash mismatch')
    if row['authority'] != AUTHORITY:
        raise EvidenceRefusal('experimental authority required')
    if row['category'] not in {'CANDIDATE', 'BASELINE', 'OPTIMUM'}:
        raise EvidenceRefusal('exact category required')
    if type(row['protocol_eligible']) is not bool or not isinstance(row['protocol_id'], str) or bool(row['protocol_id']) != row['protocol_eligible']:
        raise EvidenceRefusal('protocol must be empty exactly when ineligible')
    for key in ('run_id', 'row_id', 'arm', 'comparator', 'claim', 'backend'):
        if not _text(row[key]):
            raise EvidenceRefusal(f'{key} required')
    try:
        date = datetime.datetime.fromisoformat(row['date'].replace('Z', '+00:00'))
        if date.tzinfo is None:
            raise ValueError('timezone required')
    except (TypeError, ValueError, AttributeError) as exc:
        raise EvidenceRefusal('timezone-bearing date required') from exc
    _exact(row['identities'], IDENTITIES)
    for identity in row['identities'].values():
        _exact(identity, {'id', 'sha256'})
        if not _text(identity['id']) or not _hash(identity['sha256']):
            raise EvidenceRefusal('identities require names and digests; explicit N/A identities are hashed too')
    if schema == MEASUREMENT:
        for key in ('operator', 'metric', 'unit', 'reps_basis'):
            if not _text(row[key]):
                raise EvidenceRefusal(f'{key} required')
        if not isinstance(row['shape'], list) or not row['shape'] or not all(type(x) is int and x > 0 for x in row['shape']):
            raise EvidenceRefusal('shape required')
        if row['metric_direction'] not in {'higher_better', 'lower_better'}:
            raise EvidenceRefusal('metric direction required')
        raw = row['raw_vector']
        if type(row['repetitions']) is not int or row['repetitions'] < 1 or not isinstance(raw, list) or len(raw) != row['repetitions']:
            raise EvidenceRefusal('repetitions must count the raw vector')
        if not all(type(x) in (int, float) and math.isfinite(x) for x in raw) or type(row['value']) not in (int, float) or not math.isfinite(row['value']):
            raise EvidenceRefusal('finite metrics required')
        if row['aggregation'] != 'arithmetic_mean' or row['value'] != math.fsum(raw) / len(raw):
            raise EvidenceRefusal('value must rederive from declared raw vector')
    else:
        for key in ('fixture', 'path', 'decided_proposition'):
            if not _text(row[key]):
                raise EvidenceRefusal(f'{key} required')
        if row['claim'] != row['decided_proposition'] or row['verdict'] not in {'pass', 'fail'}:
            raise EvidenceRefusal('verifier claim must be the exact decided proposition')
        if not _hash(row['fixture_sha256']):
            raise EvidenceRefusal('fixture digest required')
        _exact(row['checker'], {'id', 'path', 'sha256'})
        if not _text(row['checker']['id']):
            raise EvidenceRefusal('checker identity required')
        if not isinstance(row['read_set'], list) or not row['read_set'] or row['read_set_sha256'] != digest(row['read_set']):
            raise EvidenceRefusal('read-set digest mismatch')
        if len({d['path'] for d in row['read_set']}) != len(row['read_set']):
            raise EvidenceRefusal('duplicate read-set file')
        if row['fixture_sha256'] not in {d['sha256'] for d in row['read_set']}:
            raise EvidenceRefusal('fixture must be in the actual read set')
        if reopen:
            _file({k: row['checker'][k] for k in ('path', 'sha256')})
            for desc in row['read_set']:
                _file(desc)
    if row['locator'] != locator(row) or row['row_id'] != digest(row['locator']):
        raise EvidenceRefusal('row locator/identity mismatch')
    if row['self_sha256'] != digest({k: v for k, v in row.items() if k != 'self_sha256'}):
        raise EvidenceRefusal('row self-hash mismatch')
    if reopen:
        _file({'path': row['attestation_path'], 'sha256': row['attestation_sha256']})
        # Attestation is the producer's original body, before self/attestation hashes.
        body = {k: v for k, v in row.items() if k not in {'self_sha256', 'attestation_path', 'attestation_sha256'}}
        if _json(row['attestation_path']) != body:
            raise EvidenceRefusal('attestation disagrees with row')
    return row


def write(directory, fields):
    """Seal a new prospective record; never overwrite a run/row identity."""
    row = dict(fields)
    row.update(producer_id=PRODUCER_ID, producer_sha256=producer_hash(), authority=AUTHORITY)
    row['locator'] = locator(row)
    row['row_id'] = digest(row['locator'])
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    att = directory / (row['row_id'] + '.attestation.json')
    dst = directory / (row['row_id'] + ('.measurement.json' if row['schema'] == MEASUREMENT else '.verifier.json'))
    body = canonical(row)
    row['attestation_path'] = str(att)
    row['attestation_sha256'] = hashlib.sha256(body).hexdigest()
    row['self_sha256'] = digest(row)
    validate(row, reopen=False)
    # Fail on repeat identity, including identical repeats: there is one writer.
    with att.open('xb') as handle:
        handle.write(body)
    validate(row)
    with dst.open('xb') as handle:
        handle.write(canonical(row))
    return row


def native_rows(path):
    data = _json(path)
    validate(data)
    return [data]


def project(row):
    validate(row)
    verifier = row['schema'] == VERIFIER
    extra = {k: row[k] for k in ('schema', 'producer_id', 'producer_sha256', 'run_id', 'row_id', 'self_sha256', 'locator', 'arm', 'comparator', 'authority', 'identities')}
    extra['promotion_authority'] = False
    extra['production_authority'] = False
    if verifier:
        extra.update({k: row[k] for k in VERIFY_FIELDS})
    else:
        extra.update(raw_vector=row['raw_vector'], aggregation=row['aggregation'])
    return dict(measurement_id=row['row_id'], metric='verdict' if verifier else row['metric'],
                value=row['verdict'] if verifier else row['value'], date=row['date'],
                category=row['category'], claim=row['claim'],
                metric_direction='higher_better' if verifier else row['metric_direction'],
                protocol_id=row['protocol_id'], reps=1 if verifier else row['repetitions'],
                reps_basis='one exact proposition checked' if verifier else row['reps_basis'],
                unit='verdict' if verifier else row['unit'], attestation_path=row['attestation_path'],
                attestation_sha256=row['attestation_sha256'], attestation_locator=canonical(row['locator']).decode(),
                attestation_present=True, attestation_verified=True,
                source_kind=row['schema'], source_class='verifier' if verifier else 'measurement',
                decided_proposition=row['decided_proposition'] if verifier else '',
                binding_kind='identity' if verifier else '', extra=extra)
