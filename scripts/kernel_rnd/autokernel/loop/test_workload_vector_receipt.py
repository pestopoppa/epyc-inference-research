"""Serving receipt retains the actual protocol for every measured leg."""
import json

from autokernel.loop import production


def test_receipt_binds_each_measured_leg_to_its_protocol(tmp_path):
    leg = {'surface': 'decode', 'baseline_record': 'p',
           'baseline_record_sha256': 'h', 'baseline_measured_at': 't',
           'effect_fraction': .1, 'effect_pct': 10., 'baseline_launches': 1,
           'champion_launches': 1, 'confidence_interval': [], 'verdict': 'x'}
    protocols = {'tg': {'model': 'M', 'server_argv': ['tg']},
                 'pp': {'model': 'M', 'server_argv': ['pp']}}
    target, evidence, _ = production._write_serving_receipt(
        tmp_path, legs={'tg': leg, 'pp': dict(leg, surface='prefill')},
        protocols=protocols, paired=True, primary='tg', champion_commit='a'*40,
        champion_build=None, frozen_commit='b'*40, frozen_label='frozen',
        production_build=None, carryover={}, source=None, bundle_name='bundle.json',
        now=lambda: '2026-10-31T00:00:00Z')
    bundle = json.loads(target.read_text())
    assert bundle['protocols'] == protocols
    assert bundle['metrics']['pp']['surface'] == 'prefill'
    assert bundle['protocol'] == protocols['tg']
    assert json.loads(evidence.read_text())['protocols'] == protocols
