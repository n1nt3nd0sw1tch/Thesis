#!/bin/bash -l
# ----------------------------------------------------------------------------
# Downloads a model into the Scratch cache, from a login node.
#
#   bash jobs/fetch.sh Qwen/Qwen3.6-27B
#
# Compute nodes have no outbound network, so anything a job needs must be here
# first. Run this once per model before submitting adaptation.sh.
# ----------------------------------------------------------------------------

MODEL="${1:?usage: bash jobs/fetch.sh <model id>}"

module unload python/miniconda3 2>/dev/null
module load python/3.11.4
source ~/ACFS/dissertation/venv/bin/activate

export HF_HOME="$HOME/Scratch/hf"
mkdir -p "$HF_HOME"

echo "fetching ${MODEL} into ${HF_HOME}"
python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1],
                         allow_patterns=['*.json', '*.safetensors', '*.txt',
                                         '*.model'])
print(f'cached at {path}')
PY

du -sh "$HF_HOME"
