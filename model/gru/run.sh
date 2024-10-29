#!/bin/bash
#SBATCH --array=0-57
#SBATCH --time=100:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-out/slurm-%A_%a.out
#SBATCH --error=slurm-out/slurm-%A_%a.out
#SBATCH --job-name=gru

python run_one.py $SLURM_ARRAY_TASK_ID
