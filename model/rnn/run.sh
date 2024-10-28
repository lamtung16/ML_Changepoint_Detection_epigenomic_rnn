#!/bin/bash
#SBATCH --array=0-1727
#SBATCH --time=72:00:00
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH --output=slurm-out/slurm-%A_%a.out
#SBATCH --error=slurm-out/slurm-%A_%a.out
#SBATCH --job-name=rnn

python run_one.py $SLURM_ARRAY_TASK_ID
