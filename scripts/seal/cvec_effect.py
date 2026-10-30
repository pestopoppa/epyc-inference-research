"""Compare pre-sampling next-token log probabilities; not raw model logits.

The fixed rule: baseline repeats must match exactly, and a shared token must
change under the vector. Missing or malformed data is inconclusive.
"""
import math


def _top(response):
    rows = response['completion_probabilities']
    if len(rows) != 1:
        raise ValueError('expected exactly one completion token')
    values = {}
    for entry in rows[0]['top_logprobs']:
        token, value = entry['id'], entry['logprob']
        if not isinstance(token, int) or token in values or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('malformed token probability')
        values[token] = float(value)
    if not values:
        raise ValueError('empty top probabilities')
    return values


def compare(baseline_a, baseline_b, steered):
    a, b, v = map(_top, (baseline_a, baseline_b, steered))
    if a != b:
        raise ValueError('baseline unstable')
    common = a.keys() & v.keys()
    delta = max((abs(a[k] - v[k]) for k in common), default=None)
    return {'vector_max_shared_logprob_delta': delta,
            'common_token_count': len(common),
            'top_token_set_changed': a.keys() != v.keys(),
            'result': 'NO_EFFECT_OBSERVED' if a == v else 'hook_effect_observed_on_fixed_fixture'}
