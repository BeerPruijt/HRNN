from config import gru_params, DATA_DIR, RESULTS_DIR
from utils.utils import construct_GRU_input, split_dataset_and_indices
from model import train_hierarchical_gru, generate_forecasts 
import pandas as pd
import torch
import numpy as np

# Load some data
cpi_data = pd.read_excel(DATA_DIR + '/public_data_april.xlsx', index_col=0)[['C000000', 'SA07']]

# Define the training data and convert to appropriate input
dataset, scaler, index_mapping = construct_GRU_input(cpi_data[['C000000']])

# Split the dataset
train_dataset, test_dataset, train_indices, test_indices = split_dataset_and_indices(dataset, index_mapping, test_pct=0.2)

# Train model
trained_model = train_hierarchical_gru(train_dataset, gru_params)

forecasts = {}
for i, tensor in enumerate(test_dataset):
    initial_sequence = [float(i) for i in list(tensor[0][0].flatten())]
    forecasted_values = generate_forecasts(trained_model, initial_sequence, 12)
    # Reshape for inverse_transform and then flatten to get back to 1D
    forecasted_values_reshaped = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1)).flatten()
    forecasts[test_indices[i]] = forecasted_values_reshaped

# Convert the forecasts dictionary to a DataFrame
forecast_df = pd.DataFrame.from_dict(forecasts, orient='index', columns=[f'+{i}' for i in range(0, 12)])

# If test_indices are datetime, the DataFrame index will be datetime
forecast_df.index = pd.to_datetime(forecast_df.index)

forecast_df.to_excel(RESULTS_DIR + '/forecasts.xlsx')

true_df = forecast_df.copy()

df_pct_change = cpi_data.pct_change(12)

for i, date in enumerate(test_indices):
    for col in range(12):
        if not np.isnan(cpi_data.loc[date+pd.DateOffset(months=col), 'C000000']):
            true_df.loc[date, f'+{col}'] = df_pct_change.loc[date+pd.DateOffset(months=col), 'C000000']
        else:
            true_df.loc[date, f'+{col}'] = np.nan

true_df.to_excel(RESULTS_DIR + '/true.xlsx')

df_RW = true_df.copy()
df_RW.loc[:, :] = 0

# Calculate the Squared Error
squared_errors_f = np.where(true_df.isna(), np.nan, (forecast_df - true_df) ** 2)
squared_errors_f = pd.DataFrame(squared_errors_f, index=forecast_df.index, columns=forecast_df.columns)
squared_errors_f.to_excel(RESULTS_DIR + '/forecasts_se.xlsx')
# Calculate the Squared Error
squared_errors_rw = np.where(true_df.isna(), np.nan, (df_RW - true_df) ** 2)
squared_errors_rw = pd.DataFrame(squared_errors_rw, index=forecast_df.index, columns=forecast_df.columns)
#squared_errors_rw.to_excel(RESULTS_DIR + '/rw_se.xlsx')

#train_data, test_data, scaler = preprocess_data(df, 'inflation_rate')

