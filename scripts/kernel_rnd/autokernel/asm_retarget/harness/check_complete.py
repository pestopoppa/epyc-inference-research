#!/usr/bin/env python3
"""rc 0 iff out/RESULT.json exists with complete=true (every run has a final record)."""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__))
try: r = json.load(open(os.path.join(D, 'out', 'RESULT.json')))
except (OSError, ValueError): sys.exit(1)
sys.exit(0 if r.get('complete') else 1)
