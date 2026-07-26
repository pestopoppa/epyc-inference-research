#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} != --execute || $# -ne 1 ]]; then
    printf 'Refusing SWE/Docker execution. Re-run with exactly: %s --execute\n' "$0" >&2
    exit 2
fi
python3 "$HERE/build_v4_scorer_package.py"
python3 - "$HERE/official_swebench_argv.json" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

package = Path(sys.argv[1]).parent
plan = json.loads(Path(sys.argv[1]).read_text())
argv = plan["argv"]
result = subprocess.run(argv, cwd=package)
if result.returncode:
    raise SystemExit(result.returncode)
validator = package / "validate_official_swebench_report.py"
raise SystemExit(subprocess.run([
    sys.executable, str(validator), "--package", str(package),
    "--report", plan["report_path"],
    "--out", str(package / "official_report.validation.json"),
], cwd=package).returncode)
PY
