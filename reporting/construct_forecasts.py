import os
import json
import pandas as pd

def check_forecasts(forecasts, expected_length):
    """
    Check if the forecasts meet the specified conditions.

    Parameters:
    forecasts (pd.Series): The forecasts to be checked.
    expected_length (int): The expected length of the forecasts.

    Raises:
    ValueError: If any of the conditions are violated.
    """
    # Check if forecasts is a pandas Series
    if not isinstance(forecasts, pd.Series):
        raise ValueError("Forecasts must be a pandas Series.")

    # Check if forecasts has a datetime index with frequency MS
    if not isinstance(forecasts.index, pd.DatetimeIndex) or forecasts.index.freq != 'MS':
        raise ValueError("Forecasts must have a datetime index with frequency MS.")

    # Check if forecasts has the expected length
    if len(forecasts) != expected_length:
        raise ValueError(f"Forecasts must have a length of {expected_length}.")

    # Check if forecasts contains any NaN values
    #if forecasts.isna().any():
    #    raise ValueError("Forecasts cannot contain NaN values.")

def convert_dict_to_json_ready(dict):
    json_ready_results = {}
    for model_spec, forecasts in dict.items():
        json_ready_forecasts = {}
        for date, forecast in forecasts.items():
            # Convert each forecast Series to a dictionary with date strings as keys
            forecast_dict = forecast.to_dict()
            # Convert date keys to string format
            forecast_dict = {date.strftime('%Y-%m-%d'): value for date, value in forecast_dict.items()}
            json_ready_forecasts[date.strftime('%Y-%m-%d')] = forecast_dict
        json_ready_results[model_spec] = json_ready_forecasts
    return json_ready_results

# Loop over models and horizons
def construct_forecasts(data, models, test_dates, save_path=None):
    """
    Constructs forecasts for a given dataset using multiple models.

    !!! IMPORTANT !!!
    The data must be only one column and the model cannot use exogenous variables.

    Parameters:
    - data (pd.DataFrame): The dataset to be used for forecasting (datetime index and only one column).
    - models (dict): A dictionary of model specifications and corresponding model functions.
    - test_dates (pd.DatetimeIndex): The dates for which forecasts are to be generated.
    - save_path (str, optional): The path to save the forecast results. Defaults to None.

    Returns:
    - results (dict): A dictionary containing the forecast results for each model and test date.

    Raises:
    - ValueError: If any of the forecast formatting conditions are violated (must be pandas Series with datetime index and frequency MS, length of 12, and no NaN values).
    - FileExistsError: If the save_path already exists.
    """
    # Define the length of the forecast window
    forecast_window = 12
    results = {}

    if save_path is not None and os.path.exists(save_path):
        raise FileExistsError(f"File already exists at {save_path}. Overwriting is not supported.")

    # Model_spec is the string that identifies the model and model_func is the function that constructs the forecast
    for model_spec, model_func in models.items():
        model_forecasts = {}
        for forecast_date in test_dates:
            # Define the last month with data that is available for this forecast
            last_month = forecast_date - pd.DateOffset(months=1)
            # Get the data up to but EXCLUDING the forecast date
            data_temp = data.loc[:last_month]
            # Construct forecasts and check if they are pandas Series with datetime index and frequency MS, length of 12, and no NaN values
            forecast_series = model_func(data_temp, forecast_window)
            check_forecasts(forecast_series, forecast_window)
            model_forecasts[forecast_date] = forecast_series
        results[model_spec] = model_forecasts

    if save_path:
        json_ready_results = convert_dict_to_json_ready(results)
        with open(save_path, 'w') as file:
            json.dump(json_ready_results, file, indent=4, sort_keys=True)

    return results