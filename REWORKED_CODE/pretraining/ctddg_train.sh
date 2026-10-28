#!/bin/bash
#SBATCH --job-name=CTDDG_Pretrain
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --output=ctddg_%j.log

set -e

cd /home/muhamed/repo/CTDDG/REWORKED_CODE/pretraining

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

PYTHON=/home/muhamed/.conda/envs/ctddg_torch/bin/python

echo "Starting CTDDG training"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

nvidia-smi

$PYTHON -u ctddg_gpu_run.py

echo "CTDDG training finished"
