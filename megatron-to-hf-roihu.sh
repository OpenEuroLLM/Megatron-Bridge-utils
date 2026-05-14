#!/bin/bash
#SBATCH --job-name=megatron-to-hf
#SBATCH --account=project_2017850
#SBATCH --partition=gpupilot
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/megatron-to-hf-%j.out
#SBATCH --error=logs/megatron-to-hf-%j.err

# Run megatron-to-hf.sh with Roihu configuration.

# Project
PROJECT="$SLURM_JOB_ACCOUNT"

# Venv base directory, setup e.g. with
# https://github.com/spyysalo/roihu-megatron-gpt/blob/main/prepare_venv.sh
VENV_DIR="/scratch/$SLURM_JOB_ACCOUNT/venv/python-pytorch-megatron-bridge"

# Paths to Megatron-Bridge and Megatron-Bridge-utils repos
BRIDGE_PATH="/scratch/$PROJECT/git_checkout/Megatron-Bridge"
UTILS_PATH="/scratch/$PROJECT/git_checkout/Megatron-Bridge-utils"

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 INPUT_PATH OUTPUT_PATH HF_MODEL TOKENIZER" >&2
    exit 1
fi

# If this script is run without sbatch, invoke with sbatch here.
if [ -z $SLURM_JOB_ID ]; then
    sbatch "$0" "$@"
    exit
fi

module purge
module load python-pytorch
source "$VENV_DIR/bin/activate"

"$UTILS_PATH/megatron-to-hf.sh" "$@" "$UTILS_PATH" "$BRIDGE_PATH"
