
import pandas as pd
import os
from utility_functions import get_acc, add_row_to_csv
import sys


datasets = [name for name in os.listdir('training_data') if os.path.isdir(os.path.join('training_data', name))]


model = sys.argv[1]
n_best_model = 4


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
    cv_df = pd.read_csv(f"model/{model}/reports/{dataset}/stat.csv")
    evaluation_df = pd.read_csv(f'training_data/{dataset}/evaluation.csv')
    fold_df = pd.read_csv(f'training_data/{dataset}/folds.csv')

    for test_fold in sorted(fold_df['fold'].unique()):
        # Filter cv_df by test_fold and test_ratio
        df_fold = cv_df[(cv_df['test_fold'] == test_fold)]
        
        # Filter evaluation dataframe by sequenceID
        eval_df = evaluation_df[evaluation_df['sequenceID'].isin(fold_df[fold_df['fold'] == test_fold]['sequenceID'])]
        
        # Get the top n models with the lowest val_loss
        top_rows = df_fold.nsmallest(n_best_model, 'val_loss')
        
        # Initialize a list to store predictions for 'llda' from the top models
        pred_list = []

        for _, row in top_rows.iterrows():
            n_layer = row['num_layers']
            layer_size = row['layer_size']
            # Load the corresponding prediction CSV for each model
            pred_df = pd.read_csv(f'model/{model}/predictions/{dataset}/{n_layer}layers_{layer_size}neurons_fold{test_fold}.csv')
            pred_df.fillna(0, inplace=True)
            # Append the 'llda' predictions to the list
            pred_list.append(pred_df['llda'])
        
        # Calculate the average of the 'llda' predictions across the models
        final_pred = sum(pred_list) / len(pred_list)
        
        # Replace the 'llda' column in pred_df with the averaged predictions
        pred_df['llda'] = final_pred
        
        # Calculate the accuracy using the averaged predictions
        acc = get_acc(eval_df, pred_df)
        
        # Save the result to the CSV file
        add_row_to_csv('acc_rate_csvs/' + dataset + '.csv', ['method', 'fold', 'acc'], [f'mlp', test_fold, acc])