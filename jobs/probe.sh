#!/bin/bash -l
# ----------------------------------------------------------------------------
# Puts the classifier on a Myriad GPU node and scores twelve fixtures.
#
#   bash jobs/fetch.sh openai/gpt-oss-safeguard-20b     once, on a login node
#   qsub jobs/probe.sh                                  one GPU on a U or V node
#   qsub -l gpu=2 -ac allow=L jobs/probe.sh             two GPUs on an L node
#   qstat -u $USER
#   cat logs/probe.o<job id>
#
# One hour and one GPU, so it schedules in minutes rather than sitting behind a
# twenty four hour reservation. It writes to results/probe/ and touches nothing
# in results/judgements/.
#
# The default asks for a U or V node, whose A100s hold 80 GiB. gpt-oss ships its
# expert weights in MXFP4, whose kernels need compute capability 9.0, and an
# A100 is 8.0, so the weights are dequantised to bfloat16 on load and the model
# wants about 48 GiB rather than the 16 GiB the model card quotes. A single
# 40 GiB A100 on an L node cannot hold it. Two of them can, with
# tensor_parallel_size 2, which probe.py sets from the visible device count.
# ----------------------------------------------------------------------------

#$ -l h_rt=1:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=UV
#$ -pe smp 4

#$ -wd /home/ucab281/Scratch/thesis
#$ -o logs/
#$ -e logs/
#$ -N probe

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

# A compute node cannot write to ACFS and has no outbound network, so the model
# cache lives in Scratch and must be populated from a login node beforehand.
export HF_HOME="$HOME/Scratch/hf"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# vLLM picks its own multiprocessing method, and the spawn method is the one
# that survives a scheduler that reaps children.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

mkdir -p logs results/probe "$HF_HOME"

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

python scripts/probe.py --backend vllm --efforts low,medium
