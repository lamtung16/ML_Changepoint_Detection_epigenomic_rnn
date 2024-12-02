#!/bin/bash
#SBATCH --array=0-35
#SBATCH --time=240:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-out/slurm-%A_%a.out
#SBATCH --error=error-out/slurm-%A_%a.out
#SBATCH --job-name=gru
#SBATCH --gpus=1

python run_one.py $SLURM_ARRAY_TASK_ID
