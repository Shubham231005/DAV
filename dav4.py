import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic time series data
days = 365 * 2
date_range = pd.date_range(start='2020-01-01', periods=days, freq='D')
trend = np.linspace(0, 15, days)
seasonality = 10 * np.sin(2 * np.pi * np.arange(days) / 365)
noise = np.random.normal(0, 2, days)

data = pd.DataFrame({'Date': date_range, 'Temperature': trend + seasonality + noise})
data.set_index('Date', inplace=True)

# Split data into train and test sets
train_size = int(len(data) * 0.9)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Plot 1: Raw Time Series Data
plt.plot(data.index, data['Temperature'], color='blue', label='Daily Temp')
plt.title('1. Raw Time Series Data (Simulated)')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.grid(True)
plt.show()

# Plot 2 & 3: ACF and PACF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(data['Temperature'], ax=ax1, lags=40)
ax1.set_title('2. Autocorrelation Function (ACF)')
plot_pacf(data['Temperature'], ax=ax2, lags=40)
ax2.set_title('3. Partial Autocorrelation (PACF)')
plt.show()

# Plot 4: Time Series Decomposition
decomp = seasonal_decompose(data['Temperature'], model='additive', period=365)
fig = decomp.plot()
fig.set_size_inches(10, 8)
plt.suptitle('4. Time Series Decomposition', y=1.02)
plt.show()

# AR Model (ARIMA(2,0,0))
print("Fitting AR Model...")
model_ar = ARIMA(train, order=(2, 0, 0))
model_ar_fit = model_ar.fit()
pred_ar = model_ar_fit.forecast(steps=len(test))

# Plot 5: AR Model Forecast
plt.plot(test.index, test, label='Actual Data', color='black')
plt.plot(test.index, pred_ar, label='AR Prediction', color='green', linestyle='--')
plt.title('5. AR Model Forecast vs Actual')
plt.legend()
plt.grid(True)
plt.show()

# MA Model (ARIMA(0,0,2))
print("Fitting MA Model...")
model_ma = ARIMA(train, order=(0, 0, 2))
model_ma_fit = model_ma.fit()
pred_ma = model_ma_fit.forecast(steps=len(test))

# Plot 6: MA Model Forecast
plt.plot(test.index, test, label='Actual Data', color='black')
plt.plot(test.index, pred_ma, label='MA Prediction', color='orange', linestyle='--')
plt.title('6. MA Model Forecast vs Actual')
plt.legend()
plt.grid(True)
plt.show()

# ARMA Model (ARIMA(2,0,2))
print("Fitting ARMA Model...")
model_arma = ARIMA(train, order=(2, 0, 2))
model_arma_fit = model_arma.fit()
pred_arma = model_arma_fit.forecast(steps=len(test))

# Plot 7: ARMA Model Forecast
plt.figure(figsize=(10, 4))
plt.plot(test.index, test, label='Actual Data', color='black')
plt.plot(test.index, pred_arma, label='ARMA Prediction', color='red', linestyle='--')
plt.title('7. ARMA Model Forecast vs Actual')
plt.legend()
plt.grid(True)
plt.show()
