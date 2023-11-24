import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

from statsmodels.tsa.ar_model import AutoReg

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

def define_ar_model(lag):
    """
    Define a closure for forecasting using an AR model with a specified lag.

    Parameters:
    lag (int): The number of lag observations to include in the model.

    Returns:
    function: A function that takes in data and a horizon and returns forecasts.
    """
    def forecast_model(data, horizon):
        """
        Forecast future values of a time series using an AR model.

        Parameters:
        data (array-like): The time series data.
        horizon (int): The number of future steps to forecast.

        Returns:
        array-like: Forecasted values.
        """
        model_fit = fit_ar_model(data, lag)
        start = len(data)
        forecast = predict_with_model(model_fit, start, horizon)
        return forecast

    return forecast_model
