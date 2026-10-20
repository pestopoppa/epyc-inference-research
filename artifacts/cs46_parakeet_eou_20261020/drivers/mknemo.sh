#!/bin/bash
set -euo pipefail
cd /mnt/raid0/llm/tmp/cs-46-20261020
export UV_CACHE_DIR=/mnt/raid0/llm/tmp/cs-46-20261020/uvcache
uv venv --python 3.12 venv-nemo
uv pip install --python venv-nemo/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchaudio
uv pip install --python venv-nemo/bin/python "nemo_toolkit[asr]>=2.5.3" soundfile
venv-nemo/bin/python -I -c "import nemo,torch;print('nemo',nemo.__version__,'torch',torch.__version__,torch.cuda.is_available())"
echo NEMO_DONE
