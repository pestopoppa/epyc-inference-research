"""Resume option wrapper (with_resume), split verbatim from serial_run.py (code-budget split)."""
from __future__ import annotations


#: Bare switch of the same family (no value): `_BOOLEAN_FLAGS` so `_without` never eats the next token.
FLOOR_REGISTRY_SWITCHES = frozenset({"--floor-allow-stale-upstream"})


_BOOLEAN_FLAGS = frozenset({*FLOOR_REGISTRY_SWITCHES, "--dry-run", "--validate-source-continuation",
                            "--validate-source-loo"})


class SerialRefused(ValueError):
    pass


def with_resume(argv, value):
    """`argv` with exactly one `--resume <value>` (run.py's checkpoint resume)."""
    if value not in ("on", "off"):
        raise SerialRefused(f"--resume must be on or off, not {value!r}")
    return [*_without(list(argv), {"--resume"}), "--resume", value]


def _without(argv, names):
    out = []
    iterator = iter(argv)
    for arg in iterator:
        flag, equals, _value = arg.partition("=")
        if flag in names:
            if not equals and flag not in _BOOLEAN_FLAGS:
                next(iterator, None)
        else:
            out.append(arg)
    return out
