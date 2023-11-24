import numpy as np
from utils import create_inout_sequences

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
