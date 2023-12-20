from data import create_dataset
import numpy as np
import pytest
import torch

# In short we expect a function that takes a numpy array and returns two torch tensors such that 1, 2, 3, 4 is mapped to ([1, 2, 3], [4]) for example

# Create fixture to test create_dataset
@pytest.fixture
def dummy_data():
    # np.array with numbers 1-20 in a column
    return np.arange(1, 21).reshape(-1, 1)

def test_create_dataset_dimensions(dummy_data):
    X, y = create_dataset(dummy_data, lookback=3)

    assert len(X) == len(y) # X and y should have the same length
    assert len(X) == 20-3 # 20 is the length of the dummy_data array and 3 is the lookback
    assert X.shape == (len(X), 3, 1) # X should have 3 columns and an additional dimension for the number of samples and the number of features
    assert y.shape == (len(y), 1) # y should have 1 column but doesn't have the dimension for the number of features
    
def test_create_dataset_X_first(dummy_data):
    X, y = create_dataset(dummy_data, lookback=3)

    # Expected values, shape and dtype
    expected_values = torch.tensor([[1], [2], [3]], dtype=torch.int32)
    expected_shape = (3, 1)
    expected_dtype = torch.int32

    # Check if the values, shape, and dtype match
    values_match = torch.equal(X[0], expected_values)
    shape_match = X[0].shape == expected_shape
    dtype_match = X[0].dtype == expected_dtype

    # Overall check
    assert values_match, "X[0] values don't match"
    assert shape_match, "X[0] shape doesn't match"
    assert dtype_match, "X[0] dtype doesn't match"

def test_create_dataset_X_last(dummy_data):
    X, y = create_dataset(dummy_data, lookback=3)

    # Expected values, shape and dtype
    expected_values = torch.tensor([[17], [18], [19]], dtype=torch.int32)
    expected_shape = (3, 1)
    expected_dtype = torch.int32

    # Check if the values, shape, and dtype match
    values_match = torch.equal(X[-1], expected_values)
    shape_match = X[-1].shape == expected_shape
    dtype_match = X[-1].dtype == expected_dtype

    # Overall check
    assert values_match, "X[-1] values don't match"
    assert shape_match, "X[-1] shape doesn't match"
    assert dtype_match, "X[-1] dtype doesn't match"


def test_create_dataset_y_first(dummy_data):
    X, y = create_dataset(dummy_data, lookback=3)

    # Expected values, shape and dtype
    expected_values = torch.tensor([4], dtype=torch.int32)
    expected_shape = (1,)
    expected_dtype = torch.int32

    # Check if the values, shape, and dtype match
    values_match = torch.equal(y[0], expected_values)
    shape_match = y[0].shape == expected_shape
    dtype_match = y[0].dtype == expected_dtype

    # Overall check
    assert values_match, f"y[0] values don't match, expected{expected_values}, got {y[0]}"
    assert shape_match, f"y[0] shape doesn't match, expected {expected_shape}, got {y[0].shape}"
    assert dtype_match, "y[0] dtype doesn't match"

def test_create_dataset_y_last(dummy_data):
    X, y = create_dataset(dummy_data, lookback=3)

    # Expected values, shape and dtype
    expected_values = torch.tensor([20], dtype=torch.int32)
    expected_shape = (1,)
    expected_dtype = torch.int32

    # Check if the values, shape, and dtype match
    values_match = torch.equal(y[-1], expected_values)
    shape_match = y[-1].shape == expected_shape
    dtype_match = y[-1].dtype == expected_dtype

    # Overall check
    assert values_match, f"y[-1] values don't match, expected{expected_values}, got {y[-1]}"
    assert shape_match, f"y[-1] shape doesn't match, expected {expected_shape}, got {y[-1].shape}"
    assert dtype_match, "y[-1] dtype doesn't match"