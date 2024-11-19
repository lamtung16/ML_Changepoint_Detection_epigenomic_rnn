import pandas as pd
import numpy as np
import sys
import os

# Load parameters
params_df = pd.read_csv("params.csv")
param_row = int(sys.argv[1])
params = params_df.iloc[param_row]

dataset    = params['dataset']
test_fold  = params['test_fold']

# create folder for predictions
os.makedirs(f'predictions/{dataset}', exist_ok=True)

# Load features, target and fold
folds_df = pd.read_csv(f'../../training_data/{dataset}/folds.csv')
target_df = pd.read_csv(f'../../training_data/{dataset}/target.csv')

# Split data into training and test sets based on the fold
train_ids = folds_df[folds_df['fold'] != test_fold]['sequenceID']
test_ids = folds_df[folds_df['fold'] == test_fold]['sequenceID']

# Prepare train sequences as arrays
target_train = target_df[target_df['sequenceID'].isin(train_ids)].iloc[:, 1:].to_numpy()


# updates intervals based on margin
def adjust_intervals(intervals, margin):
    adjusted_intervals = intervals + np.array([margin, -margin])
    mask = adjusted_intervals[:, 1] < adjusted_intervals[:, 0]
    adjusted_intervals[mask] = adjusted_intervals[mask][:, ::-1]
    return adjusted_intervals

# get best mu for each
def getting_best_mu(intervals, margin, loss_type='squared'):
    intervals = adjust_intervals(intervals, margin)
    endpoints = np.unique(intervals[np.isfinite(intervals)])
    y_min, y_max = intervals[:, 0], intervals[:, 1]
    lower_loss = np.maximum(0, y_min[:, None] - endpoints)  # Loss when mu is below y_min
    upper_loss = np.maximum(0, endpoints - y_max[:, None])  # Loss when mu is above y_max
    if loss_type == 'squared':
        lower_loss = lower_loss ** 2
        upper_loss = upper_loss ** 2
    losses = np.sum(lower_loss + upper_loss, axis=0)
    min_loss_idx = np.argmin(losses)
    mu = endpoints[min_loss_idx]
    return mu

best_mu_squared = getting_best_mu(target_train, 3)
lldas_df = pd.DataFrame({
    'sequenceID': target_df[target_df['sequenceID'].isin(test_ids)]['sequenceID'].reset_index(drop=True),
    'llda': pd.Series([best_mu_squared] * len(test_ids)).reset_index(drop=True)
})
lldas_df.to_csv(f'predictions/{dataset}/fold{test_fold}.csv', index=False)