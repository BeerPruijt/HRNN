from reporting import construct_forecasts, convert_dict_to_json_ready
import pandas as pd
import numpy as np
import pytest

def generate_dummy_forecasts(dataset, h, multiplication_factor, as_frame=False, add_nan=False):
    last_index = dataset.index[-1]
    last_value = dataset.iloc[-1]

    # Generate the date range for the new series
    new_index = pd.date_range(start=last_index + pd.DateOffset(months=1), periods=h, freq='MS')

    # Calculate the values for each index based on the last value
    new_values = (new_index.year + new_index.month + new_index.day) * last_value * multiplication_factor
    new_values = new_values.tolist()

    if add_nan:
        new_values[0] = np.nan

    # Create the new series
    dummy_series = pd.Series(new_values, index=new_index)

    if as_frame:
        dummy_series = pd.DataFrame(dummy_series, columns=['Value'])

    return dummy_series

def test_construct_forecasts():
    
    # Construct dummy data, fake models and define test dates
    dataset = pd.Series([1, 2, 3], index=pd.date_range(start='2019-01-01', periods=3, freq='MS'))
    models = {'test(1)': lambda x, y: generate_dummy_forecasts(x, y, 1), 'test(2)': lambda x, y: generate_dummy_forecasts(x, y, 2)}
    test_dates = dataset.index[-2::]

    first_indices = pd.date_range(start='2019-02-01', periods=12, freq='MS')
    second_indices = pd.date_range(start='2019-03-01', periods=12, freq='MS')

    # Expected start value for model 1: sum(date) * last value * multiplication factor 
    expected_model_1_1 = pd.Series([(idx.year + idx.month + idx.day) * 1 * 1 for idx in first_indices], index=first_indices)
    expected_model_1_2 = pd.Series([(idx.year + idx.month + idx.day) * 2 * 1 for idx in second_indices], index=second_indices)

    # Expected start value for model 2: sum(date) * last value * multiplication factor 
    expected_model_2_1 = pd.Series([(idx.year + idx.month + idx.day) * 1 * 2 for idx in first_indices], index=first_indices)
    expected_model_2_2 = pd.Series([(idx.year + idx.month + idx.day) * 2 * 2 for idx in second_indices], index=second_indices)

    # Construct the expected results dictionary
    expected_result = {}
    expected_result['test(1)'] = {pd.to_datetime('2019-02-01'): expected_model_1_1, pd.to_datetime('2019-03-01'): expected_model_1_2}
    expected_result['test(2)'] = {pd.to_datetime('2019-02-01'): expected_model_2_1, pd.to_datetime('2019-03-01'): expected_model_2_2}

    # Define the expected and actual JSON
    expected_json = convert_dict_to_json_ready(expected_result)
    actual_json = convert_dict_to_json_ready(construct_forecasts(dataset, models, test_dates, save_path=None))

    # Check that the proper sets of forecasts are generated
    assert expected_json.keys() == actual_json.keys()

    # Check that the proper months are in the sets of forecasts
    for key in expected_json.keys():
        assert expected_json[key].keys() == actual_json[key].keys()

    # Check that the proper forecasts are generated
    for key in expected_json.keys():
        assert expected_json[key] == actual_json[key]
    
# Check that the proper error is raised when the model outputs a DataFrame
def test_non_series_error():
    # Construct dummy data, fake models and define test dates
    dataset = pd.Series([1, 2, 3], index=pd.date_range(start='2019-01-01', periods=3, freq='MS'))
    models = {'test(1)': lambda x, y: generate_dummy_forecasts(x, y, 1, as_frame=True), 'test(2)': lambda x, y: generate_dummy_forecasts(x, y, 2)}
    test_dates = dataset.index[-2::]
    with pytest.raises(ValueError):
        construct_forecasts(dataset, models, test_dates, save_path=None)
