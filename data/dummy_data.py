import pandas as pd
import torch 
import numpy as np

def generate_dummy_series():

    """
    df
        Month  Passengers
    0    1949-01         112
    1    1949-02         118
    2    1949-03         132
    3    1949-04         129
    4    1949-05         121
    ..       ...         ...
    139  1960-08         606
    140  1960-09         508
    141  1960-10         461
    142  1960-11         390
    143  1960-12         432

    [144 rows x 2 columns]
    """

    # Creating a date range for months from January 1949 to December 1960
    date_range = pd.date_range(start='1949-01-01', end='1960-12-01', freq='MS')

    # Passenger data as provided
    passengers = [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
    ]

    # Creating the DataFrame
    df = pd.DataFrame({
        "Month": date_range.strftime('%Y-%m'),
        "Passengers": passengers
    })

    return df

def create_dataset(timeseries, lookback, device='cpu'):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """

    # Check if the input is a numpy ndarray
    if not isinstance(timeseries, np.ndarray):
        raise ValueError("Input is not a numpy ndarray")

    # Check if the input is 2D and the second dimension is 1
    if timeseries.ndim != 2 or timeseries.shape[1] != 1:
        raise ValueError("Input is not a 2D array with second dimension equal to 1")

    X, y = [], []
    for i in range(len(timeseries)-lookback):
        feature = timeseries[i:i+lookback] # e.g. [0, 1] for lookback=2 and i=0
        target = timeseries[i+lookback] # e.g. [2] for lookback=2 and i=0
        X.append(feature)
        y.append(target)

    return torch.tensor(np.array(X)).to(device), torch.tensor(np.array(y)).to(device)
