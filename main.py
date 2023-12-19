import pandas as pd
from model.torch_gru import forecast_using_gru
from model.torch_gru import log_diff

us_data = pd.read_csv(r"C:\Users\beerp\Data\HRNN\cpi_us_dataset.csv", index_col=0)
us_data.index = pd.to_datetime(us_data.index)
us_data = us_data[us_data.index.year > 1994]

data_temp = us_data[us_data['Category'] == 'All items']['Price']

# Convert all days to 01
data_temp.index = data_temp.index.map(lambda x: x.replace(day=1))

# Data Preparation
data = log_diff(data_temp, 1)

data.loc[:] = [i for i in range(len(data))]

all_forecasts = []

forecasts, model, loss = forecast_using_gru(data, 12, batch_size=1, num_layers=2, patience=10, num_epochs=10000, drop_prob=0.0, l2_lambda=0.00, validation_size=0.3, gradient_clipping_threshold=5, verbose=False)
all_forecasts.append(forecasts)

print(all_forecasts)