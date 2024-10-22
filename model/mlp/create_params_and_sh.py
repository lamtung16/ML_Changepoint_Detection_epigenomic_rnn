import pandas as pd
import os
from itertools import product

# CREATE PARAMETERS CSV
# Define hyperparameters
folder_path = '../../training_data'
datasets = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
param_combinations = []

# Iterate over each dataset and collect parameter combinations
for dataset in datasets:
    fold_df = pd.read_csv(os.path.join(folder_path, dataset, 'folds.csv'))
    
    test_fold = sorted(fold_df['fold'].unique())
    num_layers = [1, 2, 3]
    layer_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Create parameter grid
    combinations = product([dataset], num_layers, layer_size, test_fold)
    param_combinations.extend(combinations)

# Create DataFrame and save it into csv
params_df = pd.DataFrame(param_combinations, columns=['dataset', 'num_layers', 'layer_size', 'test_fold'])
params_df.to_csv("params.csv", index=False)

# CREATE RUN_ONE.SH
# Define job parameters
n_tasks, ncol = params_df.shape

# Create output directory for SLURM logs if it doesn't exist
output_dir = 'slurm-out'
os.makedirs(output_dir, exist_ok=True)

# Create SLURM script
run_one_contents = f"""#!/bin/bash
#SBATCH --array=0-{n_tasks-1}
#SBATCH --time=48:00:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=1
#SBATCH --output={output_dir}/slurm-%A_%a.out
#SBATCH --error={output_dir}/slurm-%A_%a.out
#SBATCH --job-name=mlp

python run_one.py $SLURM_ARRAY_TASK_ID
"""

# Write the SLURM script to a file
run_one_sh = os.path.join("run.sh")
with open(run_one_sh, "w") as run_one_f:
    run_one_f.write(run_one_contents)