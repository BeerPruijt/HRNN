import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

def define_ar_model(lag):
    # This function now returns another function (a closure) that does the actual forecasting
    def forecast_model(data, horizon):
        model = AutoReg(data, lags=lag)
        model_fit = model.fit()
        start = len(data)
        end = start + horizon - 1
        forecast = model_fit.predict(start=start, end=end)
        return forecast
    return forecast_model
