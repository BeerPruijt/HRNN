import pytest
from model.torch_gru import create_sequence_target_pairs
import numpy as np

def test_create_sequences_data_length_less_than_seq_length():
    # Test case: Data length less than seq_length
    data = [1, 2, 3]
    seq_length = 5
    try:
        create_sequence_target_pairs(data, seq_length)
        assert False, "Expected an error to be raised"
    except ValueError:
        assert True

def test_create_sequences_data_length_equal_to_seq_length_plus_one():
    # Test case: Data length equal to seq_length + 1 (for the target)
    data = [1, 2, 3, 4, 5, 6]
    seq_length = 5
    sequences, target = create_sequence_target_pairs(data, seq_length)
    expected_sequences = np.array([[1, 2, 3, 4, 5]])
    expected_target = np.array([6])

    assert np.array_equal(sequences, expected_sequences)
    assert np.array_equal(target, expected_target)

def test_create_sequences_data_length_greater_than_seq_length():
    # Test case: Data length greater than seq_length
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seq_length = 5
    expected_sequences = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]])
    expected_target = np.array([6, 7, 8, 9, 10])
    sequences, target = create_sequence_target_pairs(data, seq_length)

    assert np.array_equal(sequences, expected_sequences)
    assert np.array_equal(target, expected_target)

if __name__ == "__main__":
    pytest.main()
