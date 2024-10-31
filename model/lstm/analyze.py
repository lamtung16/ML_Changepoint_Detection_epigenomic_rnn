import pandas as pd
from glob import glob
import os
import shutil

# Define the folder path for training data
folder_path = '../../training_data'
# Get all dataset folders
datasets = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

# Initialize a list to hold all DataFrames
all_dataframes = []

# Loop through each dataset, compression type, and size
for dataset in datasets:
    for loss_type in ['square', 'linear']:
        for compress_type in ['mean', 'median']:
            for compress_size in [100, 1000, 2000]:
                # Collect all DataFrames from the specified path
                out_df_list = []
                for out_csv in glob(f"reports/{dataset}/{loss_type}/{compress_type}/{compress_size}/*.csv"):
                    out_df_list.append(pd.read_csv(out_csv))
            
                # Concatenate the DataFrames collected in this iteration
                if out_df_list:  # Check if the list is not empty
                    out_df = pd.concat(out_df_list, ignore_index=True)
                    all_dataframes.append(out_df)  # Append to the master list

# Finally, concatenate all collected DataFrames into one
if all_dataframes:  # Check if there are any DataFrames to concatenate
    final_df = pd.concat(all_dataframes, ignore_index=True)
    # Save the final concatenated DataFrame to a single stat.csv
    final_df.to_csv('stat.csv', index=False)

# Delete the whole reports folder
shutil.rmtree('reports')