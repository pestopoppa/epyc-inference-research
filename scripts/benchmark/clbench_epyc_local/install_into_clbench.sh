#!/bin/bash
# Install the epyc_local system into a CL-Bench clone. Usage: install_into_clbench.sh <clbench_repo>
set -euo pipefail
repo="${1:?usage: install_into_clbench.sh <clbench_repo>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="$repo/src/systems/epyc_local"
mkdir -p "$dest"
cp "$here/system.py" "$dest/system.py"
printf '"""epyc_local system package."""\n\nfrom .system import EpycLocalSystem\n\n__all__ = ["EpycLocalSystem"]\n' > "$dest/__init__.py"
echo "installed -> $dest"
