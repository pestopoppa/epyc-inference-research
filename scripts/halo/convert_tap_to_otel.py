#!/usr/bin/env python3
"""Compatibility CLI for closed-loop observation conversion.

The repo-readiness detector and older EPYC tooling look for a
``convert_tap_to_otel.py`` surface. The implementation lives in
``closed_loop_observation_surface.py`` because it handles benchmark, log, and
report JSONL records, not TAP alone.
"""

from __future__ import annotations

import sys

from scripts.halo.closed_loop_observation_surface import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
