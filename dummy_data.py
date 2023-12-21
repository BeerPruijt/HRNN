
import matplotlib.pyplot as plt
import pandas as pd
 
import pandas as pd
from pandas.tseries.offsets import MonthBegin
from sklearn.metrics import mean_squared_error

import numpy as np
import torch.optim as optim
import torch.utils.data as data

from data.dummy_data import generate_dummy_series, create_dataset

import torch

import torch.nn as nn

# Import minmax scaler
from sklearn.preprocessing import MinMaxScaler
 

class RnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.rnn(x)
        # Select only the last time step's output for each sequence in the batch
        x = x[:, -1, :]
        x = self.linear(x)
        return x

class DeepRnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(input_size=50, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = x[:, -1, :]  # Taking the output of the last time step
        x = self.linear(x)
        return x

def report_performance(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler, timeseries, verbose=False):
    model.eval()
    with torch.no_grad():
        # Predictions for Training Set
        train_preds = model(X_train).cpu()
        train_preds = scaler.inverse_transform(train_preds.numpy())
        train_plot = np.ones_like(timeseries) * np.nan
        train_plot[lookback:lookback + len(train_preds)] = train_preds

        # Predictions for Validation Set
        val_preds = model(X_val).cpu()
        val_preds = scaler.inverse_transform(val_preds.numpy())
        val_plot = np.ones_like(timeseries) * np.nan
        val_start_index = len(train_preds) + lookback
        val_plot[val_start_index:val_start_index + len(val_preds)] = val_preds

        # Predictions for Test Set
        test_preds = model(X_test).cpu()
        test_preds = scaler.inverse_transform(test_preds.numpy())
        test_plot = np.ones_like(timeseries) * np.nan
        test_start_index = len(train_preds) + len(val_preds) + lookback
        test_plot[test_start_index:test_start_index + len(test_preds)] = test_preds

        train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_train.cpu().numpy()), train_preds))
        val_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_val.cpu().numpy()), val_preds))
        test_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.cpu().numpy()), test_preds))

        if verbose:
            # Plotting
            plt.figure(figsize=(15, 8))
            plt.plot(timeseries_original, label='Original Data', color='blue')
            plt.plot(train_plot, label='Training Predictions', color='red')
            plt.plot(val_plot, label='Validation Predictions', color='orange')
            plt.plot(test_plot, label='Test Predictions', color='green')
            plt.legend()
            plt.show()

            # Print RMSE values
            print(f"Training RMSE: {train_rmse}")
            print(f"Validation RMSE: {val_rmse}")
            print(f"Test RMSE: {test_rmse}")

def train_val_test_split(X, y, val_size=0.1, test_size=0.1):

    assert X.shape[0] == y.shape[0]

    # train-test split for time series
    train_size = int(len(y) * (1-val_size-test_size))
    val_size = int(len(y) * val_size) # Adjust this ratio as needed
    test_size = len(y) - train_size - val_size

    # Create datasets of dimensions torch.Size([n_samples, look_back, 1]) 
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs, patience, verbose=False):
    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    best_model = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluation phase
        train_loss = evaluate_model(model, train_loader, loss_fn, device)
        val_loss = evaluate_model(model, val_loader, loss_fn, device)

        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Best Validation Loss: {best_val_loss:.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutive_no_improvement = 0
            best_model = model.state_dict()
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch + 1} as validation loss has not improved for {patience} consecutive epochs.')
            break

    # Load the best model before returning
    if best_model is not None:
        model.load_state_dict(best_model)

def evaluate_model(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            total_loss += loss_fn(y_pred, y_batch).item()

    mean_loss = total_loss / len(loader)
    return np.sqrt(mean_loss)

def generate_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, shuffle=False):

    train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=shuffle, batch_size=batch_size) 
    val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), shuffle=shuffle, batch_size=batch_size)
    test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size)

    return train_loader, val_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PARAMETERS
batch_size = 1 # higher values lead to worse performance
lookback = 4
n_epochs = 50
patience = 25
learning_rate = 0.001
verbose = True

model = DeepRnnModel().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Get the dummy data and convert to an array
df = generate_dummy_series()
df.loc[:, 'Passengers'] = df[["Passengers"]].pct_change(1) #[np.sin(i/2) for i in range(len(df))] #[i/2 for i in range(len(df))] 
timeseries_original = df.loc[df.index[1::], ["Passengers"]].values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
timeseries = scaler.fit_transform(timeseries_original)

X, y = create_dataset(timeseries, lookback=lookback, device=device)

X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, val_size=0.1, test_size=0.3)

train_loader, val_loader, test_loader = generate_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, shuffle=False)

train_model(model, train_loader, val_loader, loss_fn, optimizer, device, n_epochs, patience, verbose)

report_performance(model, X_train, y_train, X_val, y_val, X_test, y_test, scaler, timeseries, verbose=True)

