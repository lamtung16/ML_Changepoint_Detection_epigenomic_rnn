#!/bin/bash
#SBATCH --array=0-1079
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-out/slurm-%A_%a.out
#SBATCH --error=slurm-out/slurm-%A_%a.out
#SBATCH --job-name=mlp

python run_one.py $SLURM_ARRAY_TASK_ID
