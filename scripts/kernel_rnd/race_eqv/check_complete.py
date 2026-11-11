#!/usr/bin/env python3
import json, sys
try: r = json.load(open('/mnt/raid0/llm/tmp/race-eqv-gpu-r8-20261111/out/RESULT.json'))
except (OSError, ValueError): sys.exit(1)
sys.exit(0 if r.get('complete') and r.get('verdict') in ('DONE', 'INVALID_HARNESS', 'INVALID_BASELINE_UNUSABLE', 'INVALID_BASELINE_NONDETERMINISTIC', 'INVALID_NO_RESIDENCY') else 1)
