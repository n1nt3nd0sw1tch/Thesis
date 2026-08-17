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
echo "space in Scratch before the fetch:"
df -h "$HOME/Scratch" | tail -1
python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download

# jinja carries the chat template on newer models and py carries any remote
# code. Both were missing before, and a compute node with no network fails on
# the first read with an error that names the tokeniser rather than the
# download.
try:
    path = snapshot_download(sys.argv[1],
                             allow_patterns=['*.json', '*.safetensors', '*.txt',
                                             '*.model', '*.jinja', '*.py'])
except Exception as error:
    if '401' in str(error) or 'gated' in str(error).lower():
        raise SystemExit(
            f'{sys.argv[1]} is gated. Accept the licence on its model page, '
            f'then run huggingface-cli login on this node and try again.')
    raise
print(f'cached at {path}')
PY

du -sh "$HF_HOME"
