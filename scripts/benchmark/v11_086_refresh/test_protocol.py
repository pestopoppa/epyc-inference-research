"""Model-free semantic fixtures; run only through the synthetic grant job."""
import importlib.util
import json
import pathlib
import sys
import unittest

base = pathlib.Path(sys.argv.pop(1))
spec = importlib.util.spec_from_file_location('bench_protocol_086', base / 'bench_protocol_086.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ARGS = ['-m', 'model.gguf', '-t', '48', '-fa', '1', '-ctk', 'f16', '-ctv', 'f16', '-mmp', '0', '-p', '5', '-n', '32', '-d', '0,32768', '-r', '3', '-o', 'json']


class ProtocolTests(unittest.TestCase):
    def test_ordinary_accepts_all_four_cells(self):
        self.assertEqual(m.validate_bench_protocol(json.dumps([{'autokernel_hardened': False}] * 4), ARGS)['n_rows'], 4)

    def test_explicit_request_requires_seed_and_true(self):
        for seed in ('0', '18446744073709551615'):
            self.assertTrue(m.validate_bench_protocol('[{"autokernel_hardened":true}]', ARGS + ['--autokernel-harden', seed])['requested_hardened'])

    def test_underscore_alias_matches_pinned_parser(self):
        self.assertTrue(m.hardened_requested(['--autokernel_harden', '1']))

    def test_flag_looking_value_is_not_request(self):
        self.assertFalse(m.hardened_requested(['-m', '--autokernel-harden']))
        with self.assertRaises(m.BenchProtocolError):
            m.validate_bench_protocol('[{"autokernel_hardened":true}]', ['-m', '--autokernel-harden'])

    def test_unrequested_rejects_even_when_last_row(self):
        for rows in ([True], [False, True], [True, False]):
            with self.subTest(rows=rows), self.assertRaises(m.BenchProtocolError):
                m.validate_bench_protocol(json.dumps([{'autokernel_hardened': x} for x in rows]), ARGS)

    def test_explicit_request_rejects_ordinary_and_mixed(self):
        for rows in ([False], [True, False]):
            with self.subTest(rows=rows), self.assertRaises(m.BenchProtocolError):
                m.validate_bench_protocol(json.dumps([{'autokernel_hardened': x} for x in rows]), ARGS + ['--autokernel-harden', '7'])

    def test_missing_and_nonboolean_fail_closed(self):
        for row in ({}, {'autokernel_hardened': None}, {'autokernel_hardened': 0}, {'autokernel_hardened': 1}, {'autokernel_hardened': 'false'}, []):
            with self.subTest(row=row), self.assertRaises(m.BenchProtocolError):
                m.validate_bench_protocol(json.dumps([row]), ARGS)

    def test_bad_json_duplicate_and_empty_fail_closed(self):
        for text in ('', '[]', '[null]', '[{}', '[{"autokernel_hardened":false,"autokernel_hardened":true}]', '[{"autokernel_hardened":false}] garbage'):
            with self.subTest(text=text), self.assertRaises(m.BenchProtocolError):
                m.validate_bench_protocol(text, ARGS)

    def test_ambiguous_or_invalid_request_fail_closed(self):
        for args in (['--autokernel-harden'], ['--autokernel-harden=7'], ['--autokernel-harden', '-1'], ['--autokernel-harden', '18446744073709551616'], ['--autokernel-harden', 'foo'], ['unknown', 'value'], ['--', '--autokernel-harden', '7']):
            with self.subTest(args=args), self.assertRaises(m.BenchProtocolError):
                m.hardened_requested(args)

    def test_noise_prefix_supported_like_driver(self):
        self.assertFalse(m.validate_bench_protocol('ordinary log\n[{"autokernel_hardened":false}]\n', ARGS)['autokernel_hardened'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
