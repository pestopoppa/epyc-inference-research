"""Unit tests for scripts/speculative/json_schema_replay.py with a synthetic fixture in the orchestrator 42196fe27c12 layout.
Resolver tests need no llguidance; the replay test runs only when llguidance imports (set LLG_SITE) and a tokenizer.json exists."""
import hashlib, json, os, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts', 'speculative'))
import json_schema_replay as R  # noqa: E402

SCHEMA = {'type': 'object', 'properties': {'verdict': {'type': 'string', 'enum': ['accept', 'reject']}, 'score': {'type': 'integer'}},
          'required': ['verdict', 'score'], 'additionalProperties': False}
LLG_SITE = os.environ.get('LLG_SITE', '/mnt/raid0/llm/tmp/specgram3-llg/v101/lib/python3.13/site-packages')
TOKENIZER = os.environ.get('LLG_TOKENIZER', '/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw/tokenizer.json')


def canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def build(base, outputs, drop_body=(), corrupt_schema=False, raw_rows=(), corrupt_raw=False):
    os.makedirs(base + '/json_schema_store'); os.makedirs(base + '/json_schema_bodies')
    c = canon(SCHEMA); sha = hashlib.sha256(c.encode()).hexdigest()
    open(f'{base}/json_schema_store/{sha}.json', 'w').write(c + ('x' if corrupt_schema else ''))
    rawtxt = json.dumps(SCHEMA, ensure_ascii=False, separators=(',', ':'))   # original (insertion) order: verdict, score
    rsha = hashlib.sha256(rawtxt.encode()).hexdigest()
    if raw_rows:
        os.makedirs(base + '/json_schema_store/raw')
        open(f'{base}/json_schema_store/raw/{rsha}.json', 'w').write(rawtxt + ('x' if corrupt_raw else ''))
    rows = []
    for i, out in enumerate(outputs):
        rid = f'rec{i:02d}'
        prompt = f'decide {i}'
        psha = hashlib.sha256(prompt.encode()).hexdigest()
        body = None
        if rid not in drop_body:
            body = f'json_schema_bodies/{rid}.json'
            json.dump({'schema': R.BODY_SCHEMA, 'record_id': rid, 'prompt_sha256': psha, 'prompt_kind': 'prompt', 'prompt_text': prompt},
                      open(f'{base}/{body}', 'w'))
        rawf = {'schema_raw_sha256': rsha, 'schema_raw_chars': len(rawtxt)} if i in raw_rows else {}
        rows.append({'record_id': rid, 'schema': 'epyc.orchestrator.serving_call.v1', 'role': 'frontdoor', 'timings': {'predicted_n': 12},
                     'json_schema_capture': {'schema_sha256': sha, 'schema_chars': len(c), 'request_class': 'json:typed_decision' if i % 2 == 0 else 'json:review',
                                              'request_class_source': 'caller.client_class', 'output_text': out, 'output_chars': len(out),
                                              'prompt_sha256': psha, 'prompt_chars': len(prompt), 'prompt_body': body, **rawf}})
    rows.append({'record_id': 'plain', 'role': 'frontdoor'})   # no capture block: ignored
    open(base + '/serving_calls.jsonl', 'w').write('\n'.join(json.dumps(r) for r in rows) + '\n')
    return sha


class Resolve(unittest.TestCase):
    def test_resolves_all_three(self):
        with tempfile.TemporaryDirectory() as t:
            sha = build(t, ['{"verdict":"accept","score":3}', '{"verdict":"reject","score":10}'])
            rs = list(R.resolve_rows(t))
            self.assertEqual([r.record_id for r in rs], ['rec00', 'rec01'])
            r = rs[0]
            self.assertEqual((r.schema, r.output_text, r.prompt_text, r.schema_sha256, r.serving_tokens),
                             (SCHEMA, '{"verdict":"accept","score":3}', 'decide 0', sha, 12))
            self.assertEqual(r.problems, [])
            self.assertEqual(rs[1].request_class, 'json:review')

    def test_problems_reported_not_dropped(self):
        with tempfile.TemporaryDirectory() as t:
            build(t, ['{}', '{}'], drop_body=('rec01',), corrupt_schema=True)
            rs = list(R.resolve_rows(t))
            self.assertEqual(len(rs), 2)
            self.assertIn('schema_sha_mismatch', rs[0].problems)
            self.assertIsNone(rs[0].schema)
            self.assertIn('prompt_body_absent', rs[1].problems)

    def test_raw_schema_used_directly_when_present(self):
        with tempfile.TemporaryDirectory() as t:
            build(t, ['{"verdict":"accept","score":3}', '{"verdict":"reject","score":10}'], raw_rows=(0,))
            rs = list(R.resolve_rows(t))
            self.assertTrue(rs[0].schema_is_raw)
            self.assertEqual(rs[0].problems, [])
            self.assertEqual(list(rs[0].schema['properties']), ['verdict', 'score'])   # original order, not sorted
            self.assertEqual(rs[0].schema, SCHEMA)

    def test_old_row_without_raw_field_keeps_canonical(self):
        with tempfile.TemporaryDirectory() as t:
            build(t, ['{"verdict":"accept","score":3}', '{"verdict":"reject","score":10}'], raw_rows=(0,))
            r = list(R.resolve_rows(t))[1]
            self.assertFalse(r.schema_is_raw)
            self.assertEqual(r.problems, [])
            self.assertEqual(list(r.schema['properties']), ['score', 'verdict'])        # canonical store sorts

    def test_raw_sha_checked_falls_back_to_canonical(self):
        with tempfile.TemporaryDirectory() as t:
            build(t, ['{}'], raw_rows=(0,), corrupt_raw=True)
            r = list(R.resolve_rows(t))[0]
            self.assertIn('schema_raw_sha_mismatch', r.problems)
            self.assertFalse(r.schema_is_raw)
            self.assertEqual(list(r.schema['properties']), ['score', 'verdict'])

    def test_prompt_digest_checked(self):
        with tempfile.TemporaryDirectory() as t:
            build(t, ['{}'])
            p = t + '/json_schema_bodies/rec00.json'
            b = json.load(open(p)); b['prompt_text'] = 'tampered'; json.dump(b, open(p, 'w'))
            self.assertIn('prompt_sha_mismatch', list(R.resolve_rows(t))[0].problems)

    def test_class_table_unresolved_and_share(self):
        a = R.Resolved('a', 'json:x', 's', serving_tokens=30)
        b = R.Resolved('b', 'json:y', 's', serving_tokens=10)
        c = R.Resolved('c', 'json:y', 's', serving_tokens=0)
        tab = R.class_table([(a, {'ok': True, 'tokens': 10, 'forced': 4, 'single_allowed': 1, 'runs': {'2': 2}}),
                             (b, {'ok': False, 'tokens': 5}), (c, None)])
        self.assertAlmostEqual(tab['json:x']['forced_share'], 0.4)
        self.assertEqual(tab['json:x']['run_histogram'], {'2': 2})
        self.assertAlmostEqual(tab['json:x']['serving_token_share'], 0.75)
        self.assertEqual((tab['json:y']['rejected'], tab['json:y']['unresolved'], tab['json:y']['forced_share']), (1, 1, None))


@unittest.skipUnless(os.path.exists(LLG_SITE) and os.path.exists(TOKENIZER), 'llguidance site / tokenizer not present')
class Replay(unittest.TestCase):
    def test_recover_property_order(self):
        r = R.recover_property_order(json.loads(canon(SCHEMA)), {'verdict': 'accept', 'score': 1})
        self.assertEqual(list(r['properties']), ['verdict', 'score'])

    def test_raw_row_not_flagged_order_recovered(self):
        with tempfile.TemporaryDirectory() as t:
            build(t, ['{"verdict":"accept","score":3}', '{"verdict":"reject","score":10}'], raw_rows=(0,))
            out = R.run(t, TOKENIZER, LLG_SITE)
        ty = out['classes']['json:typed_decision']     # rec00 has the raw schema: direct, no heuristic
        self.assertEqual((ty['replayed_ok'], ty['order_recovered']), (1, 0))
        self.assertEqual(out['classes']['json:review']['order_recovered'], 1)   # rec01 old row: heuristic, flagged

    def test_forced_share_on_fixture(self):
        with tempfile.TemporaryDirectory() as t:
            build(t, ['{"verdict":"accept","score":3}', '{"verdict":"reject","score":10}', 'not json at all'])
            out = R.run(t, TOKENIZER, LLG_SITE)
        self.assertEqual(out['rows'], 3)
        ty, rv = out['classes']['json:typed_decision'], out['classes']['json:review']
        self.assertEqual(ty['rows'], 2)                      # rec00 ok, rec02 ('not json') rejected
        self.assertEqual((ty['replayed_ok'], ty['rejected']), (1, 1))
        self.assertEqual(ty['order_recovered'], 1)           # canonical store sorts properties: score < verdict
        self.assertEqual(rv['replayed_ok'], 1)
        for c in (ty, rv):
            self.assertGreater(c['forced_share'], 0.0)       # key names / punctuation are forced
            self.assertLess(c['forced_share'], 1.0)          # the verdict choice and the integer are not
            self.assertGreaterEqual(c['single_allowed'], 0)
        self.assertAlmostEqual(out['classes']['json:typed_decision']['serving_token_share'] + rv['serving_token_share'], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
