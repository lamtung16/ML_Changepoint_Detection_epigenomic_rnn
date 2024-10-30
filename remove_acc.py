import os
import pandas as pd
import sys

method = sys.argv[1]

folder_path = 'training_data'
datasets = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

for dataset in datasets:
    # get df
    df = pd.read_csv('acc_rate_csvs/' + dataset + '.csv')
    df = df[df['method'] != method]
    
    # overwrite csv
    df.to_csv('acc_rate_csvs/' + dataset + '.csv', index=False)