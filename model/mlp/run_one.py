import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import random
from sklearn.model_selection import train_test_split
import time
import os
from sklearn.preprocessing import MinMaxScaler
import sys


# Set random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)


# Load parameters
params_df = pd.read_csv("params.csv")
param_row = int(sys.argv[1])
params = params_df.iloc[param_row]

dataset    = params['dataset']
num_layers = params['num_layers']
layer_size = params['layer_size']
test_fold  = params['test_fold']


# create folder for predictions
os.makedirs(f'predictions/{dataset}', exist_ok=True)
os.makedirs(f'reports/{dataset}', exist_ok=True)


# Early stopping parameters
patience = 100
max_epochs = 1


# try to use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hinged Square Loss
class SquaredHingeLoss(nn.Module):
    def __init__(self, margin=1):
        super(SquaredHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, predicted, y):
        low, high = y[:, 0:1], y[:, 1:2]
        loss_low = torch.relu(low - predicted + self.margin)
        loss_high = torch.relu(predicted - high + self.margin)
        loss = loss_low + loss_high
        return torch.mean(torch.square(loss))


# MLP models
class MLPModel(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(MLPModel, self).__init__()
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, 1))  # Output layer

        self.model = nn.Sequential(*layers)  # Combine layers into a sequential model

    def forward(self, x):
        return self.model(x)


# Load features, target and fold
folds_df = pd.read_csv(f'../../training_data/{dataset}/folds.csv')
target_df = pd.read_csv(f'../../training_data/{dataset}/target.csv')
features_df = pd.read_csv(f'../../training_data/{dataset}/features.csv')


# Prepare CSV file for logging
report_path = f'reports/{dataset}/report_{param_row}.csv'
report_header = ['dataset', 'num_layers', 'layer_size', 'test_fold', 'stop_epoch', 'train_loss', 'val_loss', 'test_loss', 'time']
if not os.path.exists(report_path):
    pd.DataFrame(columns=report_header).to_csv(report_path, index=False)


# main
# Record start time
fold_start_time = time.time()

# Split data into training and test sets based on the fold
train_ids = folds_df[folds_df['fold'] != test_fold]['sequenceID']
test_ids = folds_df[folds_df['fold'] == test_fold]['sequenceID']

# Prepare train and test sequences as arrays
features_train = features_df[features_df['sequenceID'].isin(train_ids)].iloc[:, 1:].to_numpy()
target_train = target_df[target_df['sequenceID'].isin(train_ids)].iloc[:, 1:].to_numpy()
features_test = features_df[features_df['sequenceID'].isin(test_ids)].iloc[:, 1:].to_numpy()

# Normalize training features
scaler = MinMaxScaler()  # Create scaler instance
features_train = scaler.fit_transform(features_train)  # Fit on training data
features_test = scaler.transform(features_test)  # Transform test data using the same parameters

# Convert target data to tensors
target_test = torch.tensor(target_df[target_df['sequenceID'].isin(test_ids)].iloc[:, 1:].to_numpy(), dtype=torch.float32)

# Split training data into subtrain and validation sets
X_subtrain, X_val, y_subtrain, y_val = train_test_split(features_train, target_train, test_size=0.2, random_state=42)

# Move target tensors to the correct device
y_subtrain = torch.tensor(y_subtrain, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

# Initialize the MLP model, loss function, and optimizer
layer_sizes = [layer_size] * num_layers
model = MLPModel(X_subtrain.shape[1], layer_sizes).to(device)
criterion = SquaredHingeLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())

# Variables for early stopping
best_val_loss, patience_counter = float('inf'), 0
best_model_state, stop_epoch = None, 0

# Training loop
for epoch in range(max_epochs):
    model.train()
    optimizer.zero_grad()

    # Convert training input to tensor and move to device
    predictions = model(torch.tensor(X_subtrain, dtype=torch.float32).to(device))
    loss = criterion(predictions, y_subtrain)

    loss.backward()
    optimizer.step()

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(torch.tensor(X_val, dtype=torch.float32).to(device)), y_val)
        avg_test_loss = criterion(model(torch.tensor(features_test, dtype=torch.float32).to(device)), target_test.to(device))  # Ensure target_test is on the same device

    avg_train_loss = loss.item()

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss, best_model_state = val_loss.item(), model.state_dict()
        patience_counter = 0
        stop_epoch = epoch
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break

# Restore best model state for final evaluation
if best_model_state:
    model.load_state_dict(best_model_state)

# Record end time and calculate elapsed time
elapsed_time = time.time() - fold_start_time

# Log the results
report_entry = {
    'dataset': dataset,
    'num_layers': num_layers,
    'layer_size': layer_size,
    'test_fold': test_fold,
    'stop_epoch': stop_epoch,
    'train_loss': avg_train_loss,
    'val_loss': best_val_loss,
    'test_loss': avg_test_loss.item(),
    'time': elapsed_time
}
pd.DataFrame([report_entry]).to_csv(report_path, mode='a', header=False, index=False)

# Predict on the test set and save to CSV
model.eval()
pred_lldas = model(torch.tensor(features_test, dtype=torch.float32).to(device)).detach().cpu().numpy().ravel()  # Move predictions to CPU for saving
lldas_df = pd.DataFrame({'sequenceID': features_df[features_df['sequenceID'].isin(test_ids)]['sequenceID'], 'llda': pred_lldas})
lldas_df.to_csv(f'predictions/{dataset}/{num_layers}layers_{layer_size}neurons_fold{test_fold}.csv', index=False)