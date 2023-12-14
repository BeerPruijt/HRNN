from torch import nn
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim, num_layers=1, drop_prob=0):
        # This line calls the constructor of the parent class (i.e. nn.Module) and initializes it. 
        super(GRUNet, self).__init__()

        # Number of units in the hidden state of the GRU layer
        self.hidden_size = hidden_size

        # Number of recurrent layers stacked on top of each other in the GRU network
        self.num_layers = num_layers

        # Creates a GRU layer with the specified parameters
        # input_size: The number of expected features in the input x
        # hidden_size: The number of features in the hidden state h
        # num_layers: Number of recurrent layers
        # batch_first: If True, then the input and output tensors are expected as (batch_size, sequence_length, input_size)
        #              If False, then the input and output tensors are expected as (sequence_length, batch_size, input_size)
        # dropout: controls the dropout probability, a regularization technique 
        self.gru = nn.GRU(input_size=input_size, 
                          hidden_size=hidden_size, 
                          num_layers=self.num_layers, 
                          batch_first=True,
                          dropout=drop_prob)
                          
        # Creates a fully connected layer with the specified parameters
        self.fc = nn.Linear(hidden_size, output_dim)

        # Creates a ReLU activation function
        self.relu = nn.ReLU()

    # Forward pass: input x is processed through a GRU player and a fully connected layer?
    def forward(self, x, h):
        # Input x is processed through a GRU player
        out, h = self.gru(x, h)

        # The output of the last time step is fed to the fully connected layer
        out = self.fc(self.relu(out[:, -1]))

        # Return flattened (i.e. as 1 dimensional tensor) output and the updated hidden state
        return out.flatten(), h

    def init_hidden(self, batch_size, device):
        # Retreive the weight tensor from the model parameters
        weight = next(self.parameters()).data

        # Initialize the hidden state with zeros and mark it as requiring gradient computation (necessary if you want to train it using backpropagation)
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().requires_grad_().to(device)

        # Set the hidden state using Kaiming normal initialization (mean=0, variance based on fan-out of the tensor)
        nn.init.kaiming_normal_(hidden, a=0, mode='fan_out')
        return hidden
    
def create_sequence_target_pairs(data, seq_length):
    """
    Creates sequence-target pairs from the given data.

    Args:
        data (numpy.ndarray): The input data.
        seq_length (int): The length of each sequence.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array contains the input sequences, 
               and the second array contains the corresponding target values.
    """
    if seq_length > len(data):
        raise ValueError("seq_length cannot be greater than the length of data.")

    xs = []
    ys = []

    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def log_diff(data, diff_order=1):
    """Calculate the log difference of a pandas Series."""
    return np.log(data).diff(diff_order).dropna() * 100

def load_data(X, y, batch_size, shuffle=False):
    """Converts data to PyTorch tensors and creates a DataLoader."""
    X_train = torch.from_numpy(X).float().view(-1, X.shape[1], 1)  # -1 for automatic batch size adjustment
    y_train = torch.from_numpy(y).float()
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    return train_loader

def initialize_model(input_size, hidden_size, output_dim, num_layers, drop_prob):
    """Initializes the model, loss function, and optimizer."""
    model = GRUNet(input_size, hidden_size, output_dim, num_layers, drop_prob)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, loss_fn, optimizer

def train_model(model, train_loader, loss_fn, optimizer, device, num_epochs):
    """Training loop for the model, returns the trained model and loss history."""
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            h = model.init_hidden(X_batch.size(0), device)

            optimizer.zero_grad()
            output, h = model(X_batch, h)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return model, loss_history

def recursive_forecast(model, initial_sequence, forecast_horizon, device):
    """
    Performs recursive forecasting using the trained GRU model.

    Args:
        model (GRUNet): The trained GRU model.
        initial_sequence (numpy.ndarray): The initial sequence to start forecasting from.
        forecast_horizon (int): The number of future time steps to forecast.
        device (torch.device): The device (CPU or GPU) on which the model is located.

    Returns:
        numpy.ndarray: An array of forecasted values.
    """
    model.eval()  # Set the model to evaluation mode
    current_sequence = initial_sequence.copy()
    forecasts = []

    for _ in range(forecast_horizon):
        # Convert current sequence to PyTorch tensor
        sequence_tensor = torch.from_numpy(current_sequence).float().view(1, -1, 1).to(device)

        # Initialize hidden state and perform forecast
        h = model.init_hidden(1, device)
        with torch.no_grad():
            forecast, _ = model(sequence_tensor, h)
        
        # Update the sequence with the forecasted value
        forecast_value = forecast.cpu().numpy()
        forecasts.append(forecast_value)
        current_sequence = np.append(current_sequence[1:], forecast_value)

    return np.array(forecasts).flatten()

def convert_forecasts_to_series(original_series, forecasts):
    """
    Converts a list of forecasts into a Pandas Series with a DateTime index.

    Args:
        original_series (pd.Series): The original time series data.
        forecasts (list or np.ndarray): The forecasted values.

    Returns:
        pd.Series: A Pandas Series of the forecasted values with a DateTime index.
    """
    # Infer the frequency of the original series
    frequency = pd.infer_freq(original_series.index)

    if not frequency:
        raise ValueError("Unable to infer frequency of the original series. Ensure it has a consistent frequency.")

    # Generate future dates
    last_date = original_series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=len(forecasts) + 1, freq=frequency)[1:]

    # Create the forecast series
    forecast_series = pd.Series(forecasts, index=future_dates)

    return forecast_series

def forecast_using_gru(input_series, forecast_horizon, batch_size=50, num_layers=2, num_epochs=200, hidden_size=10, seq_length=4, diff_order=1):

    # Data Preparation
    data = log_diff(input_series, diff_order)
    X, y = create_sequence_target_pairs(data, seq_length)

    # Data Loading
    train_loader = load_data(X, y, batch_size)

    # Model Initialization
    model, loss_fn, optimizer = initialize_model(input_size=1, 
                                                 hidden_size=hidden_size, 
                                                 output_dim=1, 
                                                 num_layers=num_layers, 
                                                 drop_prob=0.2)

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training
    model, loss_history = train_model(model, train_loader, loss_fn, optimizer, device, num_epochs)

    # Forecasting X[-1] is the last sequence in the training set which we use to initialize the forecasting process
    forecast_values = recursive_forecast(model, X[-1], forecast_horizon, device)

    # Format as a series with datetime index
    forecasts_as_series = convert_forecasts_to_series(data, forecast_values)

    return forecasts_as_series, model, loss_history