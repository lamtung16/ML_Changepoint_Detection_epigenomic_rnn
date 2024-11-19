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
    combinations = product([dataset], test_fold)
    param_combinations.extend(combinations)

# Create DataFrame
params_df = pd.DataFrame(param_combinations, columns=['dataset', 'test_fold'])

# Check for completed rows
predictions_dir = 'predictions'  # Set the base directory for predictions
completed_indices = []

for index, row in params_df.iterrows():
    dataset = row['dataset']
    test_fold = row['test_fold']
    
    # Build the path to the prediction file
    prediction_file = os.path.join(predictions_dir, f"{dataset}/fold{test_fold}.csv")
    
    # Check if the file exists
    if os.path.exists(prediction_file):
        completed_indices.append(index)

# Remove completed rows
params_df = params_df.drop(completed_indices).reset_index(drop=True)

# Save the updated DataFrame to CSV
params_df.to_csv("params.csv", index=False)