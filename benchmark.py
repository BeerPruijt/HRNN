from config import  DATA_DIR
import pandas as pd
import numpy as np
from benchmarks import define_ar_model, define_ma_model, define_rw_model
from reporting import construct_forecasts, get_category_levels
from utils import transform_data

# Load some data
cpi_data = pd.read_excel(DATA_DIR + '/public_data_april.xlsx', index_col=0)

# Define models and horizons to test
lags = [1, 2, 3, 4]  # AR(1) - AR(4)
models = {}
for lag in lags:
    models[f'AR({lag})'] = define_ar_model(lag)
    models[f'RW({lag})'] = define_rw_model(lag)

# Define test dates1
test_dates = pd.date_range(start='2016-01-01', end='2023-04-01', freq='MS')

for level_iter in range(1, 5):

    # Load the relevant level
    columns_to_predict = list(get_category_levels(loc_codes=r"C:\Users\beerp\Data\HRNN\df_codes.xlsx")[level_iter].codes)

    for col in columns_to_predict:
        # Define the path to save the forecasts and transform the data
        path_temp = r"C:\Users\beerp\git-repos\HRNN\results\level_" + str(level_iter) + r"\forecasts_mom_" + col + ".json"
        data_temp = transform_data(cpi_data[col], log_transform=True, diff=True)
        
        # Try to construct the forecasts and save them, otherwise print the exception
        try:
            construct_forecasts(data_temp, models, test_dates, save_path=path_temp)
        except Exception as e:
            print(f"An exception occurred with column: {col}")
            print(f"Exception details: {str(e)}")
            pass
