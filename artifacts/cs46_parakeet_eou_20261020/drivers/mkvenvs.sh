#!/bin/bash
set -euo pipefail
cd /mnt/raid0/llm/tmp/cs-46-20261020
uv venv --python 3.12 venv-arm 2>&1 || uv venv venv-arm
uv pip install --python venv-arm/bin/python numpy onnxruntime soundfile scipy
venv-arm/bin/python -I -c "import onnxruntime,numpy;print('arm venv ok',onnxruntime.__version__,numpy.__version__)"
echo ARM_DONE
