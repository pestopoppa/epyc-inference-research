#!/usr/bin/env python3
"""SPEC-GRAM-JSON-1 replay reader: resolve captured json_schema serving calls and measure the grammar-forced token share offline.

Input layout (orchestrator 42196fe27c12, src/backends/serving_calls.py, flag serving_calls_json_schema_capture), all beside the log:
    serving_calls.jsonl                  rows carry a `json_schema_capture` block:
        schema_sha256, schema_chars, request_class, request_class_source, output_text, output_chars,
        prompt_sha256, prompt_chars, prompt_body ("json_schema_bodies/<record_id>.json" or None)
    json_schema_store/<schema_sha256>.json   canonical JSON (sort_keys, (',',':'), ensure_ascii False); sha256(file bytes) == name
    json_schema_bodies/<record_id>.json      {"schema": "epyc.orchestrator.serving_call_body.v1", "record_id", "prompt_sha256",
                                              "prompt_kind", "prompt_text"}; sha256(prompt_text utf-8) == prompt_sha256

resolve_rows() yields one Resolved per captured row with the schema, output and prompt, each verified against its digest; a missing or
mismatching piece is reported in `problems`, never silently dropped. replay() feeds each (schema, output) through llguidance
llg_matcher_compute_ff_tokens (the tool-call replay loop of /mnt/raid0/llm/tmp/specgram2-offline/work/replay.py, without its trim audit):
forced share of output tokens, forced-run histogram, and the single-allowed-token subset (positions the mask admits exactly one token,
the only lossless logit-free subset at T > 0). Output is RE-TOKENIZED with the given tokenizer (the serving token ids are not captured).

llguidance: the venv /mnt/raid0/llm/tmp/specgram2-offline/v ships the python files without llguidance/_lib.abi3.so (checked 2026-11-24);
json_schema_store canonicalizes with sort_keys, so stored property order is alphabetical (recover_property_order).
A working py3.13 install is /mnt/raid0/llm/tmp/specgram3-llg/v101/lib/python3.13/site-packages (llguidance 1.0.1 + numpy + tokenizers).
Pass --site to prepend either to sys.path. No model inference, no compute claim.

    python3.13 json_schema_replay.py --log <logs/serving_calls> --tokenizer <tokenizer.json> [--site <site-packages>] [--out table.json]
"""
import argparse, collections, hashlib, json, os, sys
from dataclasses import dataclass, field

BODY_SCHEMA = 'epyc.orchestrator.serving_call_body.v1'
SCHEMA_STORE_DIR = 'json_schema_store'
RAW_SCHEMA_DIR = 'json_schema_store/raw'   # orchestrator 4fea85c11524: original-order schema text, <schema_raw_sha256>.json


def _sha(b):
    return hashlib.sha256(b).hexdigest()


@dataclass
class Resolved:
    record_id: str
    request_class: str
    schema_sha256: str
    schema: dict = None
    schema_is_raw: bool = False   # True: schema is the verified original-order text (no order_recovered heuristic)
    output_text: str = None
    prompt_text: str = None
    prompt_sha256: str = None
    serving_tokens: int = None
    problems: list = field(default_factory=list)


def _serving_tokens(row):
    t = row.get('timings')
    if isinstance(t, dict):
        for k in ('predicted_n', 'tokens_predicted', 'n_predicted'):
            if isinstance(t.get(k), int):
                return t[k]
    return None


def resolve_row(row, base):
    cap = row.get('json_schema_capture')
    if not isinstance(cap, dict):
        return None
    r = Resolved(record_id=str(row.get('record_id')), request_class=cap.get('request_class') or 'unknown',
                 schema_sha256=cap.get('schema_sha256'), prompt_sha256=cap.get('prompt_sha256'),
                 serving_tokens=_serving_tokens(row))
    # schema: original-order raw text when the row carries schema_raw_sha256 (orchestrator 4fea85c11524), else the canonical store
    raw_sha = cap.get('schema_raw_sha256')
    if raw_sha:
        try:
            raw = open(os.path.join(base, RAW_SCHEMA_DIR, f'{raw_sha}.json'), 'rb').read()
            if _sha(raw) != raw_sha:
                r.problems.append('schema_raw_sha_mismatch')
            else:
                r.schema = json.loads(raw)
                r.schema_is_raw = True
                if cap.get('schema_raw_chars') not in (None, len(raw.decode('utf-8'))):
                    r.problems.append('schema_raw_chars_mismatch')
        except OSError:
            r.problems.append('schema_raw_missing')
        except ValueError:
            r.problems.append('schema_raw_unparsable')
        if r.schema is not None:
            return _resolve_rest(r, row, cap, base)
        # raw unusable (problem recorded above): fall back to the canonical store + heuristic
    sp = os.path.join(base, SCHEMA_STORE_DIR, f'{cap.get("schema_sha256")}.json')
    try:
        raw = open(sp, 'rb').read()
        if _sha(raw) != cap.get('schema_sha256'):
            r.problems.append('schema_sha_mismatch')
        else:
            r.schema = json.loads(raw)
            if cap.get('schema_chars') not in (None, len(raw.decode('utf-8'))):
                r.problems.append('schema_chars_mismatch')
    except OSError:
        r.problems.append('schema_missing')
    except ValueError:
        r.problems.append('schema_unparsable')
    return _resolve_rest(r, row, cap, base)


def _resolve_rest(r, row, cap, base):
    # output: inline
    out = cap.get('output_text')
    if isinstance(out, str):
        r.output_text = out
        if cap.get('output_chars') not in (None, len(out)):
            r.problems.append('output_chars_mismatch')
    else:
        r.problems.append('output_missing')
    # prompt: body file, relative to base
    pb = cap.get('prompt_body')
    if pb:
        try:
            body = json.load(open(os.path.join(base, pb), encoding='utf-8'))
            if body.get('schema') != BODY_SCHEMA or body.get('record_id') != row.get('record_id'):
                r.problems.append('body_identity_mismatch')
            else:
                pt = body.get('prompt_text')
                if _sha(pt.encode('utf-8', 'surrogatepass')) != cap.get('prompt_sha256'):
                    r.problems.append('prompt_sha_mismatch')
                else:
                    r.prompt_text = pt
        except (OSError, ValueError, AttributeError):
            r.problems.append('body_missing')
    else:
        r.problems.append('prompt_body_absent')
    return r


def resolve_rows(log_dir, log_name='serving_calls.jsonl'):
    """Yield Resolved for every row of log_dir/log_name that carries a json_schema_capture block (rotated *.jsonl siblings not read)."""
    with open(os.path.join(log_dir, log_name), encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if '"json_schema_capture"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            r = resolve_row(row, log_dir)
            if r is not None:
                yield r


def load_llg(site=None):
    if site:
        sys.path.insert(0, site)
    import llguidance as L
    from tokenizers import Tokenizer
    return L, Tokenizer


def make_tokenizers(tokenizer_json, site=None):
    L, Tokenizer = load_llg(site)
    return L, Tokenizer.from_file(tokenizer_json), L.LLTokenizer(open(tokenizer_json).read())


def recover_property_order(schema, value):
    """The store keeps CANONICAL JSON (sort_keys=True), which sorts every `properties` map alphabetically, while llguidance emits object
    keys in the schema's property order: the stored schema can reject the very output served under the original order. Best-effort
    recovery: reorder each `properties` map to the key order of the captured output (keys absent from the output keep sorted order,
    after). Walks type object / items; anyOf/oneOf/$ref subtrees are left as stored. Returns a new schema."""
    if isinstance(schema, dict):
        out = dict(schema)
        props = schema.get('properties')
        if isinstance(props, dict) and isinstance(value, dict):
            seen = [k for k in value if k in props]
            rest = [k for k in props if k not in value]
            out['properties'] = {k: recover_property_order(props[k], value.get(k)) for k in seen + rest}
        if isinstance(schema.get('items'), dict) and isinstance(value, list) and value:
            out['items'] = recover_property_order(schema['items'], value[0])
        return out
    return schema


def replay_one(L, tk, lt, schema, output_text, want_single=True):
    """Return dict(tokens, forced, runs, single_allowed, ok, error). Forced = tokens in compute_ff_tokens runs the output agrees with."""
    ids = tk.encode(output_text).ids
    n = len(ids)
    g = L.LLMatcher.grammar_from_json_schema(schema, defaults={'whitespace_flexible': True})
    m = L.LLMatcher(lt, g)
    if m.is_error():
        return {'ok': False, 'error': 'grammar:' + m.get_error()[:200], 'tokens': n}
    np_ = bm = None
    if want_single:
        import numpy as np_
        import llguidance.numpy as LN
        bm = LN.allocate_token_bitmask(1, lt.vocab_size)
    i = f = single = 0
    runs = collections.Counter()
    while i < n:
        ff = m.compute_ff_tokens()
        if ff and ids[i:i + len(ff)] == ff:
            m.consume_tokens(ff); f += len(ff); runs[len(ff)] += 1; i += len(ff)
            continue
        if want_single:
            LN.fill_next_token_bitmask(m, bm, 0)
            if int(np_.unpackbits(bm.view(np_.uint8)).sum()) == 1:
                single += 1
        if not m.consume_token(ids[i]):
            return {'ok': False, 'error': f'rejected at token {i}/{n}', 'tokens': n}
        i += 1
    return {'ok': bool(m.is_accepting()), 'error': None if m.is_accepting() else 'not_accepting', 'tokens': n, 'forced': f,
            'runs': dict(runs), 'single_allowed': single}


def replay_resolved(L, tk, lt, schema, output_text, schema_is_raw=False):
    """schema_is_raw: the schema is the verified original-order text, used directly (no heuristic, never flagged order_recovered).
    Otherwise (older rows): replay_one on the stored canonical schema; if rejected, once more on the output-order-recovered
    schema (flagged order_recovered)."""
    d = replay_one(L, tk, lt, schema, output_text)
    if d.get('ok') or schema_is_raw:
        return d
    try:
        val = json.loads(output_text)
    except ValueError:
        return d
    d2 = replay_one(L, tk, lt, recover_property_order(schema, val), output_text)
    if d2.get('ok'):
        d2['order_recovered'] = True
        return d2
    return d


def class_table(results):
    """results: iterable of (Resolved, replay dict | None). Per request_class table."""
    tab = {}
    for r, d in results:
        t = tab.setdefault(r.request_class, {'rows': 0, 'replayed_ok': 0, 'rejected': 0, 'unresolved': 0, 'tokens': 0, 'forced': 0,
                                             'single_allowed': 0, 'runs': collections.Counter(), 'serving_tokens': 0, 'order_recovered': 0})
        t['rows'] += 1
        t['serving_tokens'] += r.serving_tokens or 0
        if d is None:
            t['unresolved'] += 1
        elif not d.get('ok'):
            t['rejected'] += 1
        else:
            t['replayed_ok'] += 1
            t['order_recovered'] += 1 if d.get('order_recovered') else 0
            t['tokens'] += d['tokens']; t['forced'] += d['forced']; t['single_allowed'] += d['single_allowed']
            t['runs'].update({int(k): v for k, v in d['runs'].items()})
    total_serv = sum(t['serving_tokens'] for t in tab.values()) or None
    for t in tab.values():
        t['forced_share'] = t['forced'] / t['tokens'] if t['tokens'] else None
        t['single_allowed_share'] = t['single_allowed'] / t['tokens'] if t['tokens'] else None
        t['run_histogram'] = {str(k): v for k, v in sorted(t.pop('runs').items())}
        t['serving_token_share'] = t['serving_tokens'] / total_serv if total_serv else None
    return tab


def run(log_dir, tokenizer_json, site=None, limit=None):
    rows = list(resolve_rows(log_dir))[:limit]
    L, tk, lt = make_tokenizers(tokenizer_json, site)
    res = []
    for r in rows:
        d = None
        if r.schema is not None and r.output_text is not None:
            d = replay_resolved(L, tk, lt, r.schema, r.output_text, r.schema_is_raw)
        res.append((r, d))
    probs = collections.Counter(p for r in rows for p in r.problems)
    digest = _sha(open(os.path.join(log_dir, 'serving_calls.jsonl'), 'rb').read())
    return {'input_sha256': digest, 'rows': len(rows), 'problems': dict(probs), 'classes': class_table(res),
            'note': 'output re-tokenized with the given tokenizer; forced share = compute_ff_tokens runs the output agrees with'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='logs/serving_calls directory')
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--site', default=None)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    t = run(a.log, a.tokenizer, a.site, a.limit)
    s = json.dumps(t, indent=1, sort_keys=True)
    if a.out:
        open(a.out, 'w').write(s)
    print(s)
