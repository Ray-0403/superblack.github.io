from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import matplotlib.pyplot as plt
import numpy
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    return (rmse)

def MAPE(Y_true, Y_pred):
    return (numpy.mean(numpy.abs((Y_true - Y_pred) / Y_true)) * 100)


df = pd.read_csv('', names=['value'], header=0)

train = df.value[:]
test = df.value[:]

model = pm.auto_arima(train, start_p=1, start_q=1,
                      information_criterion='aic',
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=0,           # let model determine 'd'
                      seasonal=None,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
print(model.summary())

model_2 = ARIMA(train, order=(3, 0, 3))
fitted = model.fit(disp=-1)

# Forecast
fc, se, conf = fitted.forecast(12, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
# plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

rmse = format(RMSE(test, fc_series), '.4f')
mape = format(MAPE(test, fc_series), '.4f')
r2 = format(r2_score(test, fc_series), '.4f')
mae = format(mean_absolute_error(test, fc_series), '.4f')
print('RMSE:' + str(rmse) + '\n' + 'MAE:' + str(mae) + '\n' + 'MAPE:' + str(mape) + '\n' + 'R2:' + str(r2))





