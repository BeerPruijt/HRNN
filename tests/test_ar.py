import pytest
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, AutoRegResultsWrapper
from pandas.tseries.offsets import MonthBegin
from statsmodels.tools.sm_exceptions import MissingDataError

from benchmarks.ar import fit_ar_model, predict_with_model

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
    dummy_time_series_data[10] = np.nan  # Introduce a NaN value
    with pytest.raises(MissingDataError):
        fit_ar_model(dummy_time_series_data, lag=5)