"""Fail-closed protocol validation for the pinned 086v2 benchmark argv only."""
import json
import re


class BenchProtocolError(ValueError):
    pass


# Every option used by 086v2 consumes ONE following token. Values are never
# interpreted as flags. Unknown options refuse rather than guessing their arity.
_VALUE_OPTIONS = frozenset((
    '-m', '--model', '-t', '--threads', '-fa', '--flash-attn',
    '-ctk', '--cache-type-k', '-ctv', '--cache-type-v', '-mmp', '--mmap',
    '-p', '--n-prompt', '-n', '--n-gen', '-d', '--n-depth',
    '-r', '--repetitions', '-o', '--output', '-ngl', '--n-gpu-layers',
))


def hardened_requested(bench_args):
    """Parse BENCH args, excluding executable/wrappers, using pinned option arities.

    The C++ parser normalizes underscores only for --options and consumes a seed
    after --autokernel-harden. A flag-looking model value does not request hardening.
    Unsupported/ambiguous argv refuses even if a child happened to exit zero.
    """
    if not isinstance(bench_args, (list, tuple)) or not all(type(x) is str for x in bench_args):
        raise BenchProtocolError('bench argv must be a string sequence')
    requested = False
    i = 0
    while i < len(bench_args):
        option = bench_args[i]
        if option.startswith('--'):
            option = option.replace('_', '-')
        if option not in _VALUE_OPTIONS and option != '--autokernel-harden':
            raise BenchProtocolError(f'unsupported bench option: {option!r}')
        if i + 1 == len(bench_args):
            raise BenchProtocolError(f'missing value for {option}')
        if option == '--autokernel-harden':
            seed = bench_args[i + 1]
            if not re.fullmatch(r'[0-9]+', seed) or len(seed) > 20 or int(seed) > (1 << 64) - 1:
                raise BenchProtocolError('invalid hardening seed')
            requested = True
        i += 2
    return requested


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise BenchProtocolError(f'duplicate JSON key: {key}')
        obj[key] = value
    return obj


def validate_bench_protocol(text, bench_args):
    """Validate RAW rows before parse_bench drops protocol attributes.

    Return protocol metadata; mismatches are launch errors and contribute no cells.
    All rows must explicitly carry the same boolean protocol as the parsed request.
    """
    requested = hardened_requested(bench_args)
    start, end = text.find('['), text.rfind(']')
    if start < 0 or end < start:
        raise BenchProtocolError('missing bench JSON array')
    if text[end + 1:].strip():
        raise BenchProtocolError('unexpected suffix after bench JSON')
    try:
        rows = json.loads(text[start:end + 1], object_pairs_hook=_unique_object)
    except (ValueError, TypeError) as exc:
        raise BenchProtocolError(f'invalid bench JSON: {exc}') from exc
    if type(rows) is not list or not rows:
        raise BenchProtocolError('bench JSON must contain rows')
    for index, row in enumerate(rows):
        if type(row) is not dict or type(row.get('autokernel_hardened')) is not bool:
            raise BenchProtocolError(f'row {index}: missing/nonboolean autokernel_hardened')
        if row['autokernel_hardened'] != requested:
            reason = 'unrequested_hardened' if row['autokernel_hardened'] else 'requested_hardened_missing'
            raise BenchProtocolError(f'row {index}: {reason}')
    return {'requested_hardened': requested, 'autokernel_hardened': requested, 'n_rows': len(rows)}
