import collections
import json
import sqlite3

records = [json.loads(line) for line in open('rows.jsonl', encoding='utf-8')]
aggregate = json.load(open('aggregate.json', encoding='utf-8'))
db = sqlite3.connect('file:catalog-private.sqlite?mode=ro&immutable=1', uri=True)
ids = {r[0] for r in db.execute('select chunk_id from chunk')}
assert len(records) == len(ids) == aggregate['cohort_rows']
assert {r['chunk_id'] for r in records} == ids
assert len({r['chunk_id'] for r in records}) == len(records)
known = [r for r in records if r['status'] == 'known']
assert len(known) == aggregate['known_rows']
assert len(records) - len(known) == aggregate['unknown_rows']
assert sum(r['full_tokens'] >= 256 for r in known) == aggregate['at_cap_or_more']
assert sum(r['full_tokens'] > 256 for r in known) == aggregate['strictly_over_cap']
for field, value in aggregate['sums_known_only'].items():
    assert sum(r[field] for r in known) == value
assert all(0 <= r['kept_tokens'] <= min(256, r['full_tokens']) for r in known)
assert all(0 <= r['kept_covered_codepoints'] <= r['full_covered_codepoints'] <= r['raw_codepoints'] for r in known)
orphans = db.execute('select count(*) from chunk_fts f left join chunk c on c.chunk_id=f.rowid where c.chunk_id is null').fetchone()[0]
print(json.dumps({'verified_rows': len(records), 'unknown_reasons': dict(collections.Counter(r.get('reason') for r in records if r['status'] != 'known')), 'orphan_fts_rows': orphans, 'result': 'PASS'}, sort_keys=True))
