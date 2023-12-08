import pytest
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from pandas.tseries.offsets import MonthBegin
from statsmodels.tools.sm_exceptions import MissingDataError
from sklearn.metrics import mean_squared_error

from benchmarks.arma import fit_ar_model, predict_with_model, define_rw_model, define_ar_model, define_ma_model

# Create a simple time series with a known pattern
@pytest.fixture
def dummy_time_series_data():
    """
    A fixture that creates a dummy time series data set. This data set is a pandas Series
    with a datetime index (monthly frequency starting from '2000-01-01') and random values
    generated to simulate time series data. The series is 100 periods long, which means
    it spans from January 2000 to April 2008 (inclusive).
    """
    # Create a datetime index with monthly frequency starting from a specific date
    date_index = pd.date_range(start='2000-01-01', periods=100, freq='MS')

    # Create a simple time series with a known pattern
    np.random.seed(0)  # for reproducibility
    data = np.random.randn(100).cumsum()

    # Create a pandas Series with the datetime index
    time_series = pd.Series(data, index=date_index)
    return time_series


@pytest.fixture
def model_and_prediction(dummy_time_series_data):
    """
    A fixture that fits an AR model to the dummy time series data and then
    uses the model to predict a specified number of steps ahead. 

    The fixture uses the `dummy_time_series_data` fixture to get the initial
    time series data. It then fits the AR model using `fit_ar_model` and
    generates predictions using `predict_with_model`. It then returns the 
    prediction results.
    """
    lag = 5
    steps_ahead = 10
    model_fit = fit_ar_model(dummy_time_series_data, lag)
    prediction = predict_with_model(model_fit, len(dummy_time_series_data), steps_ahead)
    return prediction

# Fixture to create the dummy time series data
@pytest.fixture
def dummy_rw_data():
    # Generate a pandas Series with a datetime index and increasing values
    date_range = pd.date_range(start='2018-01-01', periods=60, freq='MS')
    series_data = pd.Series(range(60), index=date_range)
    return series_data

# Check if the model is an instance of the correct class
def test_fit_ar_model_class(dummy_time_series_data):
    lag = 5
    model_fit = fit_ar_model(dummy_time_series_data, lag)

    assert isinstance(model_fit, AutoRegResultsWrapper)

# Check if the model has the correct number of lags
def test_fit_ar_model_class(dummy_time_series_data):
    lag = 5
    model_fit = fit_ar_model(dummy_time_series_data, lag)
    
    assert max(model_fit.ar_lags) == lag

# Check if the output is a pandas Series
def test_prediction_is_series(model_and_prediction):
    assert isinstance(model_and_prediction, pd.Series)

# Check if the index of the Series corresponds to datetime
def test_prediction_index_is_datetime(model_and_prediction):
    assert isinstance(model_and_prediction.index, pd.DatetimeIndex)

# Check if the length of the Series is equal to the number of steps ahead
def test_prediction_length(model_and_prediction):
    assert len(model_and_prediction) == 10  # steps_ahead in the fixture

# Check if the Series starts at the date immediately following the last date in the input data
def test_prediction_starts_correctly(dummy_time_series_data, model_and_prediction):
    expected_start_date = dummy_time_series_data.index[-1] + MonthBegin(1)
    assert model_and_prediction.index[0] == expected_start_date

# Check if the Series spans the correct date range
def test_prediction_spans_correct_range(dummy_time_series_data, model_and_prediction):
    steps_ahead = 10
    expected_start_date = dummy_time_series_data.index[-1] + MonthBegin(1)
    expected_date_range = pd.date_range(start=expected_start_date, periods=steps_ahead, freq='MS')
    pd.testing.assert_index_equal(model_and_prediction.index, expected_date_range)

# Check if an error is raised when fitting the model with NaN values
def test_fit_ar_model_with_nan(dummy_time_series_data):
    dummy_time_series_data.iloc[10] = np.nan  # Introduce a NaN value
    with pytest.raises(MissingDataError):
        fit_ar_model(dummy_time_series_data, lag=5)

# Test to check the forecasted values and the index of the returned series
def test_rw_model(dummy_rw_data):
    # Define the lag and horizon
    lag = 1
    horizon = 3

    # Define the model using the closure
    rw_model = define_rw_model(lag)
    
    # Perform the forecasting
    forecasted_values = rw_model(dummy_rw_data, horizon)
    
    # Check if the forecasted values are a pandas Series
    assert isinstance(forecasted_values, pd.Series), "The forecast should return a pandas Series"
    
    # Check if the forecast spans the three months after the last month in the data
    expected_start_date = dummy_rw_data.index[-1] + pd.DateOffset(months=1)
    expected_dates = pd.date_range(start=expected_start_date, periods=horizon, freq='MS')
    pd.testing.assert_index_equal(forecasted_values.index, expected_dates), \
    "The forecast index should span the three months after the last month in the data"
    
    # Check if the forecasted values increase by 0 in every timestep
    assert (forecasted_values.diff().dropna() == 0).all(), \
    "The forecasted values of the RW(1) should increase by zero in every timestep"

# Test to check the RMSE of the forecasted values resulting from the forecasts is correct
def test_rw_rmse(dummy_rw_data):
    # Define the lag and horizon
    lag = 1
    horizon = 3

    # Define the model using the closure
    rw_model = define_rw_model(lag)
    
    # Perform the forecasting
    forecasted_values = rw_model(dummy_rw_data[0:-3], horizon)
    
    # Define the target and the expected RMSE
    true_values = dummy_rw_data.tail(horizon)
    expected_rmse = (1 + 4 + 9)/3 # mean(1^2 + 2^2 + 3^2)

    # Check if the indices are as expected
    assert (true_values.index == forecasted_values.index).all(), "The indices of the forecasting and the true values don't match"
    
    assert mean_squared_error(true_values, forecasted_values) == expected_rmse, "The random walk doens't yield the expected RMSE"

# Test to check if all the basic benchmark models yield the same type of outputs
def test_benchmark_forecast_types(dummy_rw_data):
    # Define the lag and horizon
    lag = 1
    horizon = 3

    # Define the model using the closure
    rw_model = define_rw_model(lag)
    ar_model = define_ar_model(lag)
    ma_model = define_ma_model(lag)

    # Perform the forecasting
    forecasts_rw = rw_model(dummy_rw_data[0:-3], horizon)
    forecasts_ar = rw_model(dummy_rw_data[0:-3], horizon)
    forecasts_ma = rw_model(dummy_rw_data[0:-3], horizon)
    
    # Check if all objects are pandas Series
    assert isinstance(forecasts_rw, pd.Series), "Forecasts of RW is not a pandas Series"
    assert isinstance(forecasts_ar, pd.Series), "Forecasts of AR not a pandas Series"
    assert isinstance(forecasts_ma, pd.Series), "Forecasts of MA not a pandas Series"

    # Compare indices
    assert forecasts_rw.index.equals(forecasts_ar.index), "Indices of RW and AR do not match"
    assert forecasts_rw.index.equals(forecasts_ma.index), "Indices of RW and MA do not match"