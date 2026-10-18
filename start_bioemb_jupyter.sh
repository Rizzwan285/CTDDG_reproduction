#!/bin/bash
#SBATCH --job-name=bioemb-jupyter
#SBATCH --partition=gpu04
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:59:59
#SBATCH --output=jupyter-bioemb-%j.log

module load anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bioemb

echo "Compute node: $(hostname)"
echo "Python: $(which python)"
echo "Conda environment: $CONDA_PREFIX"
nvidia-smi

jupyter lab --no-browser --ip=0.0.0.0 --port=8890
