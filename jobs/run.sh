#!/bin/bash -l
# ----------------------------------------------------------------------------
# Runs one stage of the pipeline on a Myriad A100 node.
#
#   cd ~/Scratch/thesis
#   qsub jobs/run.sh                                  generation, default model
#   qsub -v MODEL=Qwen/Qwen3.6-27B jobs/run.sh        generation, another model
#   qsub -v STAGE=judge jobs/run.sh                   scoring, every reply
#   qstat -u $USER
#   tail -f logs/thesis.o<job id>
#
# Results append one line at a time and a rerun skips whatever is already there,
# so a job that hits the wall clock can be submitted again and will carry on.
# ----------------------------------------------------------------------------

# Wall clock, memory per core, cores, and a single GPU. The L request asks for
# an A100: the V100 nodes lack bfloat16 and cannot hold a model of this size.
#$ -l h_rt=24:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=L
#$ -pe smp 8

# Somewhere to write. Scratch is writable from a compute node; ACFS is not.
#$ -wd /home/ucab281/Scratch/thesis
#$ -o logs/
#$ -e logs/
#$ -N thesis

STAGE="${STAGE:-generate}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
REPLICATES="${REPLICATES:-3}"

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

mkdir -p logs results "$HF_HOME"

# ----------------------------------------------------------------------------
# Report what this job is
# ----------------------------------------------------------------------------

echo "job      ${JOB_ID:-interactive} on $(hostname)"
echo "started  $(date '+%Y-%m-%d %H:%M:%S')"
echo "stage    ${STAGE}"
echo "model    ${MODEL}, ${REPLICATES} replicates"
echo "python   $(python --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------

if [ "${STAGE}" = "judge" ]; then
    python scripts/evaluate.py --backend vllm
else
    python scripts/run.py generate \
        --model "${MODEL}" \
        --backend vllm \
        --replicates "${REPLICATES}"
fi

echo
echo "finished $(date '+%Y-%m-%d %H:%M:%S')"
