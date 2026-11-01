#!/usr/bin/env python3
"""Prospective paired scorer adapter; accepts only complete canonical per-item scores."""
import hashlib
import json
import sys
from pathlib import Path
SNAPSHOT=Path('/mnt/raid0/llm/tmp/codex3-cs8-stt3-transcribe-20261030')
CATALOG_SHA='74f4c45bacf0ea2da83d9869e5b0f0e33fa8a5a2d3043081c0da7d2f4c18b8a2'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def scorer():
    catalog=SNAPSHOT/'SCORER-SHA256SUMS'
    if sha(catalog)!=CATALOG_SHA: raise ValueError('scorer catalog drift')
    for line in catalog.read_text().splitlines():
        h,n=line.split('  ',1)
        if sha(SNAPSHOT/n)!=h: raise ValueError('scorer member drift')
    sys.path.insert(0,str(SNAPSHOT/'scorer-snapshot'))
    from scripts.speech.stt_bakeoff.scorer import paired_bootstrap_delta
    return paired_bootstrap_delta

def score_utterance_chunks(row, chunks, expected_chunk_hashes):
    # Catalog verification must precede importing the frozen scorer.
    scorer()
    from scripts.speech.stt_bakeoff.scorer import score_utterance
    if not expected_chunk_hashes or len(chunks) != len(expected_chunk_hashes) or [c.get('chunk_index') for c in chunks] != list(range(len(chunks))):
        raise ValueError('missing or unordered chunk')
    if [c.get('pcm_sha256') for c in chunks] != expected_chunk_hashes:
        raise ValueError('chunk PCM mismatch')
    if any(c.get('status') not in ('complete','failure') for c in chunks):
        raise ValueError('unknown exposure cannot be scored')
    if any(c.get('status') == 'complete' and not isinstance(c.get('text'),str) for c in chunks):
        raise ValueError('completed chunk missing text')
    failed=any(c['status']=='failure' for c in chunks)
    texts=['' for c in chunks] if failed else [c['text'] for c in chunks]
    raw={'utterance_id':row['clip_id'],'pcm_sha256':row['pcm_sha256'],
         'reference_raw':row['reference_raw'],'hypothesis_chunks':texts,
         'chunk_indices':list(range(len(texts))),'duration_s':row['frames']/16000,
         'language':row['language'].split('_')[0],'set_id':'fleurs-validation',
         'reference_track':'standard','content_stratum':'clean_read',
         'decode_status':'error' if failed else 'ok'}
    return score_utterance(raw,normaliser_id='stt_norm/v1',segmentation_mode='vad_30s',repetition_envelope=None)

def compare(expected, baseline, candidate):
    if set(expected)!={'en','it'}: raise ValueError('expected en and it')
    pair=scorer(); out={}
    for lang in ('en','it'):
        ids=expected[lang]
        if not ids or len(ids)!=len(set(ids)): raise ValueError('invalid denominator')
        a=baseline[lang];b=candidate[lang]
        x={r['utterance_id']:r for r in a}; y={r['utterance_id']:r for r in b}
        if len(x)!=len(a) or len(y)!=len(b) or set(x)!=set(ids) or set(y)!=set(ids): raise ValueError('incomplete or duplicate denominator')
        for i in ids:
            if any(z.get('class') in ('VOID','unknown_marker','numeral_uncovered') for z in (x[i],y[i])): raise ValueError('unknown scored case')
            if any(x[i].get(k)!=y[i].get(k) for k in ('pcm_sha256','reference_tokens','reference_normalised')): raise ValueError('unpaired input')
            if x[i].get('pcm_sha256') is None or x[i].get('reference_normalised') is None: raise ValueError('missing pairing evidence')
        out[lang]=pair([x[i]['errors'] for i in ids],[y[i]['errors'] for i in ids],
                       [x[i]['reference_tokens'] for i in ids],resamples=4000,seed=146064415)
    return {'schema':'cs8-stt8-paired-screen/v1','languages':out,
            'stt8_gate': 'PASS' if all(out[l]['interval_pp'][1] <= 1.0 for l in ('en','it')) else 'FAIL',
            'other_stt5_advancement_criteria':'not assessed','promotion':False}

def compare_fleurs_validation(baseline, candidate):
    path=Path('/mnt/raid0/llm/data/stt-cs8-public-20261030/fleurs-validation-pcm/manifest.json')
    if sha(path)!='1bf2544d1d02d8f72b1a57d293c1e0869953aae31c3c526220b7965df09b536e':
        raise ValueError('validation manifest drift')
    rows=json.loads(path.read_bytes())['records']
    expected={short:[r['clip_id'] for r in rows if r['language']==full]
              for short,full in (('en','en_us'),('it','it_it'))}
    if (len(expected['en']),len(expected['it'])) != (394,391):
        raise ValueError('unexpected validation denominator')
    return compare(expected,baseline,candidate)
