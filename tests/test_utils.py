import numpy as np
from utils import create_inout_sequences, transform_data, inverse_transform_data
import pandas as pd
import pytest

# Sample data for testing
@pytest.fixture
def sample_time_series():
    dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
    data = pd.Series(np.random.rand(36), index=dates)
    return data

def test_log_transformation(sample_time_series):
    transformed_data = transform_data(sample_time_series, log_transform=True)
    assert np.allclose(np.log(sample_time_series), transformed_data)

def test_regular_differencing(sample_time_series):
    transformed_data = transform_data(sample_time_series, diff=True)
    assert np.allclose(sample_time_series.diff().dropna(), transformed_data)

def test_seasonal_differencing(sample_time_series):
    transformed_data = transform_data(sample_time_series, seasonal_diff=True, seasonal_lag=12)
    assert np.allclose(sample_time_series.diff(periods=12).dropna(), transformed_data)

def test_non_series_input():
    with pytest.raises(ValueError):
        transform_data([1, 2, 3, 4], log_transform=True)

def test_incorrect_frequency():
    dates = pd.date_range(start='2020-01-01', periods=24, freq='D')  # Daily frequency
    data = pd.Series(np.random.rand(24), index=dates)
    with pytest.raises(ValueError):
        transform_data(data, log_transform=True)

def test_reverse_log_transformation(sample_time_series):
    log_transformed = np.log(sample_time_series)
    reversed_data = inverse_transform_data(log_transformed, log_transform=True)
    assert np.allclose(sample_time_series, reversed_data, atol=1e-5)

def test_reverse_regular_differencing(sample_time_series):
    initial_value = sample_time_series.iloc[0]
    diffed = sample_time_series.diff().dropna()
    reversed_data = inverse_transform_data(diffed, initial_value=initial_value, diff=True)
    assert np.allclose(sample_time_series, reversed_data, atol=1e-5)

def test_error_for_missing_indices_in_additional_data(sample_time_series):
    seasonal_lag = 12
    # Create additional_data missing some required indices
    missing_indices_data = sample_time_series.head(seasonal_lag - 1)  # Intentionally one index short

    seasonal_diffed = sample_time_series.diff(periods=seasonal_lag).dropna()
    with pytest.raises(ValueError):
        inverse_transform_data(seasonal_diffed, seasonal_diff=True, seasonal_lag=seasonal_lag, additional_data=missing_indices_data)

def test_reverse_seasonal_differencing(sample_time_series):
    data_for_reversal = sample_time_series.head(12)
    seasonal_diffed = sample_time_series.diff(periods=12).dropna()
    reversed_data = inverse_transform_data(seasonal_diffed, seasonal_diff=True, seasonal_lag=12, additional_data=data_for_reversal)
    assert np.allclose(reversed_data, sample_time_series.iloc[12:], atol=1e-5)

def test_reverse_non_series_input():
    with pytest.raises(ValueError):
        inverse_transform_data([1, 2, 3, 4], log_transform=True)

def test_insufficient_additional_data(sample_time_series):
    seasonal_diffed = sample_time_series.diff(periods=12).dropna()
    insufficient_additional_data = sample_time_series.head(5)  # Less than 12
    with pytest.raises(ValueError):
        inverse_transform_data(seasonal_diffed, seasonal_diff=True, seasonal_lag=12, additional_data=insufficient_additional_data)

def test_create_inout_sequences_2d_array():
    # Test data for 2D NumPy array
    input_array = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

    sequence_length = 3

    # Expected output for 2D array
    expected_output_array = [
        (np.array([[1], [2], [3]]), np.array([[4]])),
        (np.array([[2], [3], [4]]), np.array([[5]])),
        (np.array([[3], [4], [5]]), np.array([[6]])),
        (np.array([[4], [5], [6]]), np.array([[7]])),
        (np.array([[5], [6], [7]]), np.array([[8]])),
        (np.array([[6], [7], [8]]), np.array([[9]])),
        (np.array([[7], [8], [9]]), np.array([[10]]))
    ]

    # Call the function with 2D NumPy array input
    result_array = create_inout_sequences(input_array, sequence_length)

    # Assert (check) that the result matches the expected output for array input
    for (seq_res, label_res), (seq_exp, label_exp) in zip(result_array, expected_output_array):
        assert np.array_equal(seq_res, seq_exp) and np.array_equal(label_res, label_exp)

    # Additional check for data types in the result
    assert all(isinstance(seq, np.ndarray) and isinstance(label, np.ndarray) for seq, label in result_array)

def test_create_inout_sequences_list():
    # Test data
    input_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sequence_length = 3

    # Expected output
    expected_output = [
        ([1, 2, 3], [4]),
        ([2, 3, 4], [5]),
        ([3, 4, 5], [6]),
        ([4, 5, 6], [7]),
        ([5, 6, 7], [8]),
        ([6, 7, 8], [9]),
        ([7, 8, 9], [10])
    ]

    # Call the function
    result = create_inout_sequences(input_data, sequence_length)

    # Assert (check) that the result matches the expected output
    assert result == expected_output
