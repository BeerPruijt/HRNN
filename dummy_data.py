
import matplotlib.pyplot as plt
import pandas as pd
 
import pandas as pd
from pandas.tseries.offsets import MonthBegin

import numpy as np
import torch.optim as optim
import torch.utils.data as data

from data.dummy_data import generate_dummy_series, create_dataset

import torch

import torch.nn as nn

# Import minmax scaler
from sklearn.preprocessing import MinMaxScaler
 
#The output of nn.LSTM() is a tuple. The first element is the generated hidden states, one for each time step of the input. The second element is the LSTM cell’s memory and hidden states, which is not used here.
#The LSTM layer is created with option batch_first=True because the tensors you prepared is in the dimension of (window sample, time steps, features) and where a batch is created by sampling on the first dimension.

class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.RNN(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

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

# PARAMETERS
batch_size = 1 # higher values lead to worse performance
lookback = 3
n_epochs = 50
patience = 10
learning_rate = 0.001
verbose = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get the dummy data and convert to an array
df = generate_dummy_series()
df.loc[:, 'Passengers'] = [np.sin(i/2) for i in range(len(df))] #[i/2 for i in range(len(df))] #
timeseries_original = df[["Passengers"]].values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
timeseries = scaler.fit_transform(timeseries_original)

# train-test split for time series
train_size = int(len(timeseries) * 0.57)
val_size = int(len(timeseries) * 0.1) # Adjust this ratio as needed
test_size = len(timeseries) - train_size - val_size

train = timeseries[:train_size]
val = timeseries[train_size:train_size + val_size]
test = timeseries[train_size + val_size:]

# Create datasets of dimensions torch.Size([n_samples, look_back, 1]) 
X_train, y_train = create_dataset(train, lookback=lookback, device=device)
X_val, y_val = create_dataset(val, lookback=lookback, device=device) 
X_test, y_test = create_dataset(test, lookback=lookback, device=device) 

model = AirModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# shuffling seems to not do anything
train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size) 
val_loader = data.DataLoader(data.TensorDataset(X_val, y_val), shuffle=True, batch_size=batch_size)
test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), shuffle=True, batch_size=batch_size)

train_model(model, train_loader, val_loader, loss_fn, optimizer, device, n_epochs, patience, verbose)

with torch.no_grad():
    # shift train predictions for plotting
    train_plot = np.ones_like(timeseries) * np.nan
    y_pred = model(X_train)
    y_pred = y_pred[:, -1, :]
    train_preds = scaler.inverse_transform(model(X_train)[:, -1, :].cpu())
    val_preds = scaler.inverse_transform(model(X_train)[:, -1, :].cpu())
    train_plot[lookback:train_size] = train_preds
    # shift test predictions for plotting
    test_plot = np.ones_like(timeseries) * np.nan
    test_preds = scaler.inverse_transform(model(X_test)[:, -1, :].cpu())
    test_plot[train_size+val_size+lookback:len(timeseries)] = test_preds
    # plot
    plt.plot(timeseries_original, c='b')
    plt.plot(train_plot, c='r')
    plt.plot(test_plot, c='g')
    plt.show()


