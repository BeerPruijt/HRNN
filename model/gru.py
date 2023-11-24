import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def define_gru_model(sequence_length):
    def revert_growth_rates_to_levels(last_level, growth_rates):
        levels = [last_level]
        for growth_rate in growth_rates:
            new_level = levels[-1] * (1 + growth_rate)
            levels.append(new_level)
        return levels[1:]

    def forecast_model(data_input, forecast_horizon):
        
        data = data_input.copy()
        data = data.pct_change(1).dropna()

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = scaler.fit_transform(np.array(data).reshape(-1, 1))

        # Function to create sequences for forecasting
        def create_sequences(data, sequence_length):
            xs, ys = [], []
            for i in range(len(data) - sequence_length):
                x = data[i:(i + sequence_length)]
                y = data[i + sequence_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        # Create sequences
        X, y = create_sequences(data_normalized, sequence_length)

        # Build the GRU model
        model = Sequential()
        model.add(GRU(units=10, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(GRU(units=10))
        model.add(Dense(1))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min', restore_best_weights=True)

        # Train the model with early stopping
        model.fit(X, y, epochs=10000, batch_size=50, verbose=1, callbacks=[early_stopping])

        # Forecasting
        forecast = []
        current_batch = data_normalized[-sequence_length:].reshape(1, sequence_length, 1)
        for i in range(forecast_horizon):
            current_pred = model.predict(current_batch)[0]
            forecast.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        # Inverse transform to original scale
        forecast = scaler.inverse_transform(forecast).flatten()

        # Revert growth rates to levels
        last_known_level = data_input.iloc[-1]
        forecast_level = revert_growth_rates_to_levels(last_known_level, forecast)

        # Create a pandas Series with datetime index
        start_date = pd.to_datetime(data.index[-1]) + pd.DateOffset(months=1)
        dates = pd.date_range(start=start_date, periods=forecast_horizon, freq='MS')
        forecast_series = pd.Series(forecast_level, index=dates)

        return forecast_series

    return forecast_model