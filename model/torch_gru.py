from torch import nn
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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

def construct_dataloader(X, y, batch_size, shuffle=False):
    """Convert the data to dataloaders, which can generate batches of the data for training and testing."""
    # Convert data to PyTorch tensors (numpy arrays but with additional functionality)
    X = torch.from_numpy(X).float().view(-1, X.shape[1], 1)
    y = torch.from_numpy(y).float()

    # Create DataLoader objects for training and testing data
    data = TensorDataset(X, y)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)  

    return loader

def mse_with_l2_loss(outputs, targets, model, l2_lambda=0.0):
    """Computes the MSE loss with optional L2 regularization."""
    # Standard loss (e.g., MSE)
    loss = torch.nn.MSELoss()(outputs, targets)
    
    # L2 regularization term
    l2_reg = torch.tensor(0.).to(targets.device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += l2_lambda * l2_reg
    
    return loss

def initialize_model(input_size, hidden_size, output_dim, num_layers, drop_prob):
    """Initializes the model, loss function, and optimizer."""
    model = GRUNet(input_size, hidden_size, output_dim, num_layers, drop_prob)
    loss_fn = mse_with_l2_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, loss_fn, optimizer

def train_model_with_early_stopping(model, train_loader, val_loader, loss_fn, optimizer, device, num_epochs, patience, gradient_clipping_threshold=1.0, l2_lambda=0.0, verbose=True):
    """
    Training loop for the model with early stopping.
    
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        loss_fn: The loss function to optimize.
        optimizer: The optimizer for model parameter updates.
        device (str): Device to run the model on ('cpu' or 'cuda').
        num_epochs (int): Maximum number of training epochs.
        patience (int): Number of consecutive epochs with no validation loss improvement to trigger early stopping.

    Returns:
        model (nn.Module): The trained model.
        loss_history (list): List of average training losses for each epoch.
    """
    # Initialize the loss history, best validation loss, and the number of consecutive epochs with no improvement in validation loss
    loss_history_training = []
    loss_history_validation = []
    best_val_loss = np.inf
    consecutive_no_improvement = 0
    best_model = None

    for epoch in range(num_epochs):
        # Training loss (use train loader and enable backpropagation of the gradients)
        train_loss = evaluate_model(model, train_loader, loss_fn, optimizer, device, enable_backpropagation=True, gradient_clipping_threshold=gradient_clipping_threshold, l2_lambda=l2_lambda)
        loss_history_training.append(train_loss)

        # Validation loss (use validation loader and disable backpropagation of the gradients), we also disable L2 regularization (i.e. simply evaluate the model based on MSE loss)
        val_loss = evaluate_model(model, val_loader, loss_fn, optimizer, device, enable_backpropagation=False, gradient_clipping_threshold=gradient_clipping_threshold, l2_lambda=0.0)
        loss_history_validation.append(val_loss)

        if verbose:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Best Validation Loss: {best_val_loss:.4f}')

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            consecutive_no_improvement = 0
            # Update the best model
            best_model = model.state_dict()
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= patience:
            if verbose:
                print(f'Early stopping at epoch {epoch + 1} as validation loss has not improved for {patience} consecutive epochs.')
            break

    # Load the best model state
    if best_model is not None:
        model.load_state_dict(best_model)
            
    loss_history = {'training': loss_history_training, 'validation': loss_history_validation}

    return model, loss_history

def evaluate_model(model, data_loader, loss_fn, optimizer, device, enable_backpropagation=True, gradient_clipping_threshold=1.0, l2_lambda=0.0):
    """
    Training or evaluation loop to calculate loss on a dataset (training is when backpropagation is applied when evaluating).
    
    Args:
        model (nn.Module): The neural network model to train or evaluate.
        data_loader (DataLoader): DataLoader for the dataset.
        loss_fn: The loss function used for training or evaluation.
        optimizer: The optimizer for model parameter updates (only used if enable_backpropagation=True).
        device (str): Device to run the model on ('cpu' or 'cuda').
        enable_backpropagation (bool): Whether to enable gradient computation and parameter updates.

    Returns:
        avg_loss (float): Average loss on the dataset.
    """
    model.train() if enable_backpropagation else model.eval()
    total_loss = 0

    # Enable or disable gradient calculation based on the 'enable_backpropagation' flag
    with torch.set_grad_enabled(enable_backpropagation):
        for X_batch, y_batch in data_loader:
            # Initialize hidden state and move to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            h = model.init_hidden(X_batch.size(0), device)

            # Calculate loss for the batch            
            output, h = model(X_batch, h)
            loss = loss_fn(output, y_batch, model, l2_lambda=l2_lambda)

            # Backpropagation and parameter updates only if training is enabled
            if enable_backpropagation:
                # Clear the gradients otherwise they accumulate from step to step
                optimizer.zero_grad()
                # After computing the forward pass and caluclating the loss, calculates parameter gradients w.r.t. the loss
                loss.backward()
                # Limit the norm of the gradients to prevent numerical instability (i.e. exploding gradients)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_threshold)
                # Update the parameters using the clipped gradients
                optimizer.step()

            # Accumulate the loss
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

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

def forecast_using_gru(input_series, forecast_horizon, batch_size=50, num_layers=2, num_epochs=10000, hidden_size=10, seq_length=4, gradient_clipping_threshold=1.0, patience=10, random_state=None, validation_size=0.3, drop_prob=0.0, l2_lambda=0.0, verbose=True):

    data = input_series.copy()

    # Scale data
    scaler = MinMaxScaler()
    data.loc[:] = scaler.fit_transform(data.values.reshape(-1, 1)).ravel()

    X, y = create_sequence_target_pairs(data, seq_length)

    # Split the data into training and validation sets (the validation set is used for the early stopping mechanism)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=validation_size, shuffle=True, random_state=random_state)

    # Format the data as loaders
    train_loader = construct_dataloader(X_train, y_train, batch_size, shuffle=False)
    validation_loader = construct_dataloader(X_validation, y_validation, batch_size, shuffle=False)

    # Model Initialization
    model, loss_fn, optimizer = initialize_model(input_size=1, 
                                                 hidden_size=hidden_size, 
                                                 output_dim=1, 
                                                 num_layers=num_layers, 
                                                 drop_prob=drop_prob)

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training
    model, loss_history = train_model_with_early_stopping(model, train_loader, validation_loader, loss_fn, optimizer, device, num_epochs, patience, gradient_clipping_threshold=gradient_clipping_threshold, l2_lambda=l2_lambda, verbose=verbose)

    # Forecasting X[-1] is the last sequence in the training set which we use to initialize the forecasting process
    last_sequence_scaled = scaler.transform(X[-1].reshape(-1, 1)).ravel()
    forecast_values = recursive_forecast(model, last_sequence_scaled, forecast_horizon, device)

    # Format as a series with datetime index
    forecasts_as_series = convert_forecasts_to_series(data, forecast_values)

    # Inverse transform the forecasts
    forecasts_as_series.loc[:] = scaler.inverse_transform(forecasts_as_series.values.reshape(-1, 1)).ravel()

    return forecasts_as_series, model, loss_history