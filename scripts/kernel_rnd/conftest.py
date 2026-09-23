"""Make the kernel_rnd suites location-independent (P7.3).

Some autokernel modules import ``scripts.benchmark`` / ``scripts.kernel_rnd`` by
their repo-root package path, which only resolves when the repository root is on
``sys.path``. pytest run from the repo root has that for free; run from
``scripts/kernel_rnd`` it does not, so put it there explicitly.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
