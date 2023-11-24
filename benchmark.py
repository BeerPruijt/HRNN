from config import  DATA_DIR, RESULTS_DIR
import pandas as pd
import numpy as np
from benchmarks import define_ar_model
from model import define_gru_model

# Load some data
cpi_data = pd.read_excel(DATA_DIR + '/public_data_april.xlsx', index_col=0)[['C000000', 'SA07']]
data = cpi_data['C000000'].dropna()

# Define models and horizons to test
lags = [1, 2, 3, 4]  # AR(1) - AR(4)
models = {}
for lag in lags:
    models[f'AR({lag})'] = define_ar_model(lag)
    models[f'GRU({lag})'] = define_gru_model(lag)

# Define test dates
test_dates = pd.date_range(start='2016-01-01', end='2023-04-01', freq='MS')

# Store results
results = []

# Loop over models and horizons
for last_month in test_dates:
    data_temp = data.loc[:last_month]
    for model_spec in models.keys():
        forecast = models[model_spec](data_temp, 12)
        results.append({
            'model_specification': model_spec,
            'month': last_month,
            'forecast_values': forecast.tolist()  # Assuming forecast is a numpy array or similar
        })

pd.DataFrame(results).to_excel(RESULTS_DIR + '/forecasts.xlsx')
