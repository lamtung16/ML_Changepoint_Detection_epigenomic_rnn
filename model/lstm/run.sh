#!/bin/bash
#SBATCH --array=0-3455
#SBATCH --time=24:00:00
#SBATCH --mem=6GB
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-out/slurm-%A_%a.out
#SBATCH --error=error-out/slurm-%A_%a.out
#SBATCH --job-name=lstm

python run_one.py $SLURM_ARRAY_TASK_ID
