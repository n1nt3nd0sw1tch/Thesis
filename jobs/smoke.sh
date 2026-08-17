#!/bin/bash -l
# ----------------------------------------------------------------------------
# Puts four real benchmark prompts through one model on a Myriad node and
# prints what came back.
#
# Run these in order. Do not start at the seventy billion parameter model: it
# fails for one of six reasons and the error will not say which.
#
#   1. prove the environment, one GPU, about four minutes
#        bash jobs/fetch.sh Qwen/Qwen2.5-1.5B-Instruct
#        qsub jobs/smoke.sh
#
#   2. prove tensor parallelism, two GPUs, about ten minutes
#        bash jobs/fetch.sh openai/gpt-oss-safeguard-20b
#        qsub -v MODEL=openai/gpt-oss-safeguard-20b,GPUS=2 jobs/smoke.sh
#
#   3. the one you asked for, 141 GB of weights
#        huggingface-cli login          once, the repository is gated
#        bash jobs/fetch.sh meta-llama/Llama-3.3-70B-Instruct
#        qsub -v MODEL=meta-llama/Llama-3.3-70B-Instruct,GPUS=2 jobs/smoke.sh
#
#   qstat -u $USER
#   tail -f logs/smoke.o<job id>
#   cat results/probe/generate-*.md
#
# Three things to know before step 3.
#
# Memory. Llama 3.3 70B is about 141 GB at bfloat16. An L node holds four 40 GB
# A100s, so 160 GB, which leaves under 20 GB for the KV cache once the weights
# are in and needs all four cards. A U or V node holds four 80 GB A100s, so two
# cards give the same 160 GB and four give 320 GB with room to spare. Two on a
# U or V node is the sensible request and is what GPUS=2 asks for below.
#
# Disk. 141 GB has to sit in Scratch before the job starts, because a compute
# node has no outbound network. Check the quota with `lquota` on a login node
# first, and clear the cache with `rm -rf ~/Scratch/hf/hub/models--*` if it is
# tight. This is the step most likely to stop you.
#
# Access. meta-llama repositories are gated. Accept the licence on the model
# page and run `huggingface-cli login` on a login node, or fetch.sh will fail
# with a 401 that reads like a network fault.
# ----------------------------------------------------------------------------

#$ -l h_rt=2:00:00
#$ -l mem=32G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 8

#$ -wd /home/ucab281/Scratch/thesis
#$ -o logs/
#$ -e logs/
#$ -N smoke

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TOKENS="${TOKENS:-1024}"

# The GPU count has to be requested at submission rather than set here, because
# the scheduler reads the directives above before this line runs. Pass it with
# -v GPUS=2 and qsub -l gpu=2, or use the ready made lines in the header.

# ----------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------

# The miniconda module conflicts with the python module the venv was built
# against, so it is unloaded first. Without the python module the venv fails
# with a missing libpython3.11.so.1.0.
module unload python/miniconda3 2>/dev/null
module load python/3.11.4
module load cuda/12.2.2/gnu-10.2.0

source ~/ACFS/dissertation/venv/bin/activate

# A compute node cannot write to ACFS and has no outbound network, so the cache
# lives in Scratch and must be populated from a login node beforehand.
export HF_HOME="$HOME/Scratch/hf"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# vLLM picks its own multiprocessing method and spawn is the one that survives
# a scheduler that reaps children.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# If a multi GPU load hangs at "Loading safetensors" with no progress, the peer
# to peer transport is the usual cause on PCIe nodes. Uncomment this and try
# again before assuming the model does not fit.
# export NCCL_P2P_DISABLE=1

mkdir -p logs results/probe "$HF_HOME"

# ----------------------------------------------------------------------------
# Report what this job is
# ----------------------------------------------------------------------------

echo "job      ${JOB_ID:-interactive} on $(hostname)"
echo "started  $(date '+%Y-%m-%d %H:%M:%S')"
echo "model    ${MODEL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h "$HOME/Scratch" | tail -1
echo

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

python scripts/probe.py \
    --task generate \
    --model "${MODEL}" \
    --tokens "${TOKENS}" \
    --temperature 1.0 \
    --utilisation 0.92 \
    --length 4096

echo
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
