#!/usr/bin/env python3
"""Freeze first 20 English validation entries in source order, without model exposure."""
import hashlib
import json
from pathlib import Path

SOURCE = Path('/mnt/raid0/llm/data/stt-cs8-public-20261030/fleurs-validation-pcm/manifest.json')
EXPECTED = '1bf2544d1d02d8f72b1a57d293c1e0869953aae31c3c526220b7965df09b536e'
OUT = Path('/mnt/raid0/llm/tmp/codex3-cs8-stt8-20261101/rt1-cohort.json')
def digest(b): return hashlib.sha256(b).hexdigest()
raw = SOURCE.read_bytes()
if digest(raw) != EXPECTED: raise SystemExit('source manifest mismatch')
rows = [r for r in json.loads(raw)['records'] if r['language'] == 'en_us'][:20]
if len(rows) != 20: raise SystemExit('insufficient English items')
out = []
for i, r in enumerate(rows):
    data = Path(r['wav_path']).read_bytes()
    if digest(data) != r['wav_sha256']: raise SystemExit('wav mismatch')
    out.append({'id': f'en_{i:02d}', 'clip_id':r['clip_id'], 'path':r['wav_path'], 'sha256':r['wav_sha256'], 'pcm_sha256':r['pcm_sha256'], 'ordinal':r['ordinal']})
obj = {'schema':'cs8-rt1-cohort/v1', 'selection':'first 20 en_us validation rows in source order; no redraw', 'source_manifest':str(SOURCE), 'source_manifest_sha256':EXPECTED, 'utterances':out}
if OUT.exists(): raise SystemExit('cohort already exists; do not overwrite freeze')
payload=(json.dumps(obj,sort_keys=True,indent=2)+'\n').encode()
with OUT.open('xb') as f: f.write(payload)
print('cohort_sha256', digest(payload), 'count', len(out))
