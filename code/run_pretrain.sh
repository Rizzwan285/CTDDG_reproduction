#!/bin/bash
#SBATCH --job-name=ctddg-pretrain
#SBATCH --partition=gpu04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/muhamed/repo/CTDDG/code/logs/ctddg_%j.log
#SBATCH --error=/home/muhamed/repo/CTDDG/code/logs/ctddg_%j.err

set -Eeuo pipefail

# Navigate to the project directory
cd /home/muhamed/repo/CTDDG/code

# Activate Conda safely
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate ctddg_torch
set -u

# Log execution metadata
echo "========================================"
echo "CTDDG pretraining started: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "========================================"

# GPU diagnostics
nvidia-smi

# Start training and stream logs immediately.
# Set RUN_TRAINING=1 to actually train.
# Keep FORCE_RESTART=0 to allow checkpoint resumption.
export CTDDG_RUN_TRAINING=1
export CTDDG_FORCE_RESTART=1

python -u pretraining_torch.py
status=$?

echo "========================================"
echo "CTDDG pretraining finished: $(date)"
echo "Exit status: $status"
echo "========================================"

exit "$status"