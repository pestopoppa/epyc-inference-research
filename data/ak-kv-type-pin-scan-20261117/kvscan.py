"""Report-only scan (AK-BENCH-KV-TYPE-PIN): stored keep rows, and the KV type their llama-bench evidence ran.
Opens each store read-only. Never re-grades anything."""
import glob, json, sqlite3, sys

stores = sorted(glob.glob('/mnt/raid0/llm/autokernel/campaigns/*/store*/experiments.db')) + ['/mnt/raid0/llm/autokernel/loop-memory/experiments.db']
out = []
for db in stores:
    try:
        c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    except sqlite3.Error as e:
        out.append({'store': db, 'error': str(e)}); continue
    cols = [r[1] for r in c.execute('pragma table_info(experiments)')]
    status_col = 'status' if 'status' in cols else None
    rec_col = next((x for x in ('record', 'payload', 'row_json', 'body') if x in cols), None)
    if not status_col:
        out.append({'store': db, 'error': 'no status column', 'cols': cols}); continue
    q = f"select attempt_id, target_surface, {rec_col or 'NULL'} from experiments where {status_col} in ('kept','keep')"
    rows = c.execute(q).fetchall()
    per = {'store': db, 'keeps': len(rows), 'kv_recorded': {}, 'kv_absent_implicit_f16': 0, 'surfaces': {}}
    for aid, surf, rec in rows:
        per['surfaces'][surf] = per['surfaces'].get(surf, 0) + 1
        kv = None
        if rec:
            try:
                body = json.loads(rec)
                comp = body.get('comparison') or {}
                kv = (comp.get('kv_type') if isinstance(comp, dict) else None) or body.get('kv_type')
            except (ValueError, AttributeError):
                pass
        if kv:
            per['kv_recorded'][kv] = per['kv_recorded'].get(kv, 0) + 1
        else:
            per['kv_absent_implicit_f16'] += 1
    per['record_column'] = rec_col
    out.append(per)
json.dump(out, sys.stdout, indent=1)
