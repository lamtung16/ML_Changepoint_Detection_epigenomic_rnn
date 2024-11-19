import pandas as pd
import os
from utility_functions import get_acc, add_row_to_csv
import sys


datasets = [name for name in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', name))]


model = 'constant'


def add_row_to_csv(file_path, columns, row):
    # Check if the file exists, and if so, read it to check for duplicates
    try:
        existing_df = pd.read_csv(file_path)
    except FileNotFoundError:
        existing_df = pd.DataFrame(columns=columns)

    # Check if the combination of method and fold already exists
    method = row[0]
    fold = row[1]
    if any((existing_df['method'] == method) & (existing_df['fold'] == fold)):
        print(f"Warning: Entry for method '{method}' and fold '{fold}' already exists in '{file_path}'. Row not added.")
        return

    # Append the new row to the existing DataFrame
    new_row = pd.DataFrame([row], columns=columns)
    updated_df = pd.concat([existing_df, new_row], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    updated_df.to_csv(file_path, index=False)


for dataset in datasets:
    # Load the necessary CSV files
    fold_df = pd.read_csv(f'training_data/{dataset}/folds.csv')
    evaluation_df = pd.read_csv(f'training_data/{dataset}/evaluation.csv')

    for test_fold in sorted(fold_df['fold'].unique()):
        pred_df = pd.read_csv(f'model/{model}/predictions/{dataset}/fold{test_fold}.csv')
        eval_df = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] == test_fold]['sequenceID'])]
        
        # Calculate the accuracy using the averaged predictions
        acc = get_acc(eval_df, pred_df)
        
        # Save the result to the CSV file
        add_row_to_csv('acc_rate_csvs/' + dataset + '.csv', ['method', 'fold', 'acc'], [f'{model}', test_fold, acc])