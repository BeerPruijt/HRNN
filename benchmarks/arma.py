import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

from utils import transform_data, inverse_transform_data

def fit_ar_model(data, lag):
    """
    Fit an Autoregressive (AR) model to the provided time series data.

    Parameters:
    data (array-like): The time series data.
    lag (int): The number of lag observations to include in the model.

    Returns:
    AutoRegResults: The fitted AR model.
    """
    model = AutoReg(data, lags=lag)
    model_fit = model.fit()
    return model_fit

def predict_with_model(model_fit, start, horizon):
    """
    Make predictions using a fitted AR model.

    Parameters:
    model_fit (AutoRegResults): The fitted AR model.
    start (int): The start point for prediction in the data series.
    horizon (int): The number of steps to predict.

    Returns:
    array-like: Predicted values.
    """
    end = start + horizon - 1
    forecast = model_fit.predict(start=start, end=end)
    return forecast

def define_ar_model(lag, log_transform=False, diff=False, seasonal_diff=False, seasonal_lag=12):
    """
    Define a closure for forecasting using an AR model with a specified lag and data transformations.

    Parameters:
    lag (int): The number of lag observations to include in the model.
    log_diff (bool): Apply log differencing if True.
    diff (bool): Apply regular differencing if True.
    seasonal_diff (bool): Apply seasonal differencing if True.
    seasonal_lag (int): The lag for seasonal differencing.

    Returns:
    function: A function that takes in data and a horizon and returns forecasts.
    """
    def forecast_model(data, horizon):
        """
        Forecast future values of a time series using an AR model with data transformations.

        Parameters:
        data (array-like): The time series data.
        horizon (int): The number of future steps to forecast.

        Returns:
        array-like: Forecasted values.
        """
        # Apply transformations
        transformed_data = transform_data(data, log_transform=log_transform, diff=diff, seasonal_diff=seasonal_diff, seasonal_lag=seasonal_lag)

        # Fit the AR model
        model_fit = fit_ar_model(transformed_data, lag)

        # Predict
        start = len(transformed_data)
        forecast = predict_with_model(model_fit, start, horizon)

        # Reverse transformations on forecasted data
        initial_value = data.iloc[-seasonal_lag] if seasonal_diff else data.iloc[-1]
        forecast = inverse_transform_data(forecast, initial_value=initial_value, log_transform=log_transform, diff=diff, seasonal_diff=seasonal_diff, seasonal_lag=seasonal_lag, additional_data=data)

        return forecast

    return forecast_model

def fit_ma_model(data, lag):
    """
    Fit a Moving Average (MA) model to the provided time series data.

    Parameters:
    data (array-like): The time series data.
    lag (int): The number of lag errors to include in the model.

    Returns:
    ARIMAResults: The fitted MA model.
    """
    model = ARIMA(data, order=(0, 0, lag))
    model_fit = model.fit()
    return model_fit

def predict_with_model(model_fit, start, horizon):
    """
    Make predictions using a fitted MA model.

    Parameters:
    model_fit (ARIMAResults): The fitted MA model.
    start (int): The start point for prediction in the data series.
    horizon (int): The number of steps to predict.

    Returns:
    array-like: Predicted values.
    """
    end = start + horizon - 1
    forecast = model_fit.predict(start=start, end=end)
    return forecast

def define_ma_model(lag, log_transform=False, diff=False, seasonal_diff=False, seasonal_lag=12):
    """
    Define a closure for forecasting using an MA model with a specified lag and data transformations.

    Parameters:
    lag (int): The number of lag errors to include in the model.
    log_diff (bool): Apply log differencing if True.
    diff (bool): Apply regular differencing if True.
    seasonal_diff (bool): Apply seasonal differencing if True.
    seasonal_lag (int): The lag for seasonal differencing.

    Returns:
    function: A function that takes in data and a horizon and returns forecasts.
    """
    def forecast_model(data, horizon):
        """
        Forecast future values of a time series using an MA model with data transformations.

        Parameters:
        data (array-like): The time series data.
        horizon (int): The number of future steps to forecast.

        Returns:
        array-like: Forecasted values.
        """
        # Apply transformations
        transformed_data = transform_data(data, log_transform=log_transform, diff=diff, seasonal_diff=seasonal_diff, seasonal_lag=seasonal_lag)

        # Fit the MA model
        model_fit = fit_ma_model(transformed_data, lag)

        # Predict
        start = len(transformed_data)
        forecast = predict_with_model(model_fit, start, horizon)

        # Reverse transformations on forecasted data
        initial_value = data.iloc[-seasonal_lag] if seasonal_diff else None
        forecast = inverse_transform_data(forecast, initial_value=initial_value, log_transform=log_transform, diff=diff, seasonal_diff=seasonal_diff, seasonal_lag=seasonal_lag, additional_data=data)

        return forecast

    return forecast_model

def fit_rw_model(data, lag):
    """
    Fit a Random Walk (RW) model with specified lag to the provided time series data.

    Parameters:
    data (pd.Series): The time series data.
    lag (int): The number of lag observations to include in the model.

    Returns:
    pd.Series: A 'model' which is essentially the last 'lag' observations.
    """
    if len(data) < lag:
        raise ValueError("The length of the data must be greater than the specified lag.")
    return data[-lag:]

def predict_with_rw_model(model, start_date, horizon):
    """
    Make predictions using a fitted RW model.

    Parameters:
    model (pd.Series): The last 'lag' observations from the fitted RW model.
    start_date (datetime): The start date for the forecast period.
    horizon (int): The number of steps to predict.

    Returns:
    pd.Series: Predicted values, which are the average of the 'lag' observations.
    """
    average_value = model.mean()
    dates = pd.date_range(start=start_date, periods=horizon, freq='MS')
    forecast = pd.Series([average_value] * horizon, index=dates)
    return forecast

def define_rw_model(lag):
    """
    Define a closure for forecasting using an RW model with a specified lag.

    Parameters:
    lag (int): The number of lag observations to include in the model.

    Returns:
    function: A function that takes in data and a horizon and returns forecasts.
    """
    def forecast_model(data, horizon):
        """
        Forecast future values of a time series using an RW model.

        Parameters:
        data (pd.Series): The time series data.
        horizon (int): The number of future steps to forecast.

        Returns:
        pd.Series: Forecasted values.
        """
        # Fit the RW model
        model = fit_rw_model(data, lag)

        # Predict
        start_date = data.index[-1] + pd.DateOffset(months=1)
        forecast = predict_with_rw_model(model, start_date, horizon)

        return forecast

    return forecast_model