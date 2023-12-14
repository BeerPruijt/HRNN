from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, Subset
import numpy as np
import pandas as pd

def preprocess_data(df):

    mask = df.isna()
    data = df.pct_change(12)
    data[mask] = np.nan
    data = data.dropna()

    # Keep track of the indices that remain after dropping NaN values
    remaining_indices = data.index.to_list()
    
    data = data.values.astype(float)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data)


    return data_normalized, scaler, remaining_indices

def create_inout_sequences(input_data, tw):
    """
    Create input-output sequences from time series data.

    This function takes a time series dataset and generates sequences of data points
    (lags) to be used as inputs for predicting the next data point in the series.
    Each input sequence consists of 'tw' (for time window) number of consecutive data points, and the
    output is the data point immediately following the input sequence.

    Parameters:
    input_data (list or array-like): The time series data from which to generate sequences.
    tw (int): The sequence length or the number of time steps/lags to use for prediction.
              This determines how many previous data points are used to predict the next.

    Returns:
    list: A list of tuples, where each tuple contains an input sequence (train_seq) of
          length 'tw' and the corresponding label (train_label) which is the next data point
          in the time series.

    Example:
    Given input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and tw = 3,
    the function will return [([1, 2, 3], [4]), ([2, 3, 4], [5]), ..., ([7, 8, 9], [10])].

    This is useful for preparing data for time series forecasting models where the goal
    is to predict future values based on observed historical data.
    """
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


def split_dataset_and_indices(dataset, indices, test_pct):
    """
    Splits a TensorDataset and a corresponding list of indices into training and test sets.

    Parameters:
    dataset (TensorDataset): The dataset to be split.
    indices (list): A list of indices corresponding to the dataset.
    test_pct (float): The percentage of the dataset to be used as the test set.

    Returns:
    tuple: A tuple containing four elements:
        - TensorDataset: The training set.
        - TensorDataset: The test set.
        - list: The indices corresponding to the training set.
        - list: The indices corresponding to the test set.
    """
    total_size = len(dataset)
    test_size = int(total_size * test_pct)
    train_size = total_size - test_size

    # Split the dataset
    train_dataset = TensorDataset(*[t[:train_size] for t in dataset.tensors])
    test_dataset = TensorDataset(*[t[train_size:] for t in dataset.tensors])

    # Split the indices
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return train_dataset, test_dataset, train_indices, test_indices

def construct_GRU_input(df):

    sequence_length = 4

    # Make no changes to the original data
    df_output = df.copy()

    # Preprocess the specified column, scaler is used to rescale the forecasts back to the original scale
    data_normalized, scaler, remaining_indices = preprocess_data(df_output)

    # Generate sequences of type (np.array([[4], [5], [6]]), np.array([[7]])) and subsequely split them in input and target tensors
    result_array = create_inout_sequences(data_normalized, sequence_length)
    inputs = [torch.tensor(s[0], dtype=torch.float32) for s in result_array]
    targets = [torch.tensor(s[1], dtype=torch.float32) for s in result_array]

    # This will combine the inputs and targets into a single dataset
    dataset = TensorDataset(torch.stack(inputs), torch.stack(targets))

    # Create a mapping from TensorDataset index to original DataFrame index
    index_mapping = remaining_indices[sequence_length:]

    return dataset, scaler, index_mapping


def transform_data(data, log_transform=False, diff=False, seasonal_diff=False, seasonal_lag=12):
    """
    Apply log transformation, regular differencing, and/or seasonal differencing to the data.

    Parameters:
    data (Pandas Series): The time series data with a datetime index and 'MS' frequency.
    log_transform (bool): Apply log transformation if True.
    diff (bool): Apply regular differencing if True.
    seasonal_diff (bool): Apply seasonal differencing if True.
    seasonal_lag (int): The lag for seasonal differencing.

    Returns:
    Pandas Series: Transformed data.
    """
    if not isinstance(data, pd.Series) or not pd.infer_freq(data.index) == 'MS':
        raise ValueError("Data must be a Pandas Series with a datetime index and 'MS' frequency.")

    if log_transform:
        data = np.log(data)

    if diff:
        data = data.diff().dropna()

    if seasonal_diff:
        data = data.diff(periods=seasonal_lag).dropna()

    return data.asfreq('MS')

def inverse_transform_data(forecasted_data, initial_value=None, diff=False, log_transform=False, seasonal_diff=False, seasonal_lag=12, additional_data=None):
    """Parameters:
    forecasted_data (numpy.ndarray): The forecasted data after transformations.
    initial_value (float): The initial value of the original series before differencing.
    diff (bool): Indicates if differencing was applied.
    log_transform (bool): Indicates if log transformation was applied.
    seasonal_diff (bool): Indicates if seasonal differencing was applied.

    Returns:
    numpy.ndarray: Data after reversing the transformations.
    """
    if seasonal_diff:
        raise NotImplementedError("Seasonal differencing is not yet supported.")
    
    if diff and log_transform and initial_value is not None:
        output_data = initial_value*np.exp(np.cumsum(forecasted_data))
    else:
        raise NotImplementedError("Only log_diff reversal is supported.")
    
    return output_data

