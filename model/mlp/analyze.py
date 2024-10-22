import pandas as pd
from glob import glob
import os

folder_path = '../../training_data'
datasets = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

for dataset in datasets:
    out_df_list = []
    for out_csv in glob(f"reports/{dataset}/*.csv"):
        out_df_list.append(pd.read_csv(out_csv))
    out_df = pd.concat(out_df_list, ignore_index=False)
    out_df.to_csv(f"reports/{dataset}/stat.csv", index=False)