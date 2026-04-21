import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Load Dataset
data = pd.read_csv("weather.csv")
series = data["Temp3pm"].dropna()
series.index = range(1, len(series)+1)

# Step 2: Visualize Data
plt.figure(figsize=(10,5))
plt.plot(series)
plt.title("Original Time Series (Temp3pm)")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()

# Step 3: Check Stationarity (ADF Test)
result = adfuller(series)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Step 4: Differencing
diff_series = series.diff().dropna()

plt.figure(figsize=(10,5))
plt.plot(diff_series)
plt.title("Differenced Series")
plt.show()

# Step 5: ACF and PACF
plot_acf(diff_series)
plt.show()

plot_pacf(diff_series)
plt.show()

# Step 6: Train-Test Split
train_size = int(len(series)*0.8)
train = series[:train_size]
test = series[train_size:]

# Step 7: Fit ARIMA Model
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit()
print(model_fit.summary())

# Step 8: Forecast
predictions = model_fit.forecast(steps=len(test))

# Step 9: RMSE Accuracy
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

# Step 10: PEV Accuracy
actual = np.array(test)
predicted = np.array(predictions)
pev = np.abs((actual - predicted) / actual) * 100
avg_pev = np.mean(pev)
accuracy = 100 - avg_pev
print("Average PEV (%):", avg_pev)
print("Model Accuracy (%):", accuracy)

# Step 11: Plot Forecast vs Actual
plt.figure(figsize=(10,5))
plt.plot(train, label="Training Data")
plt.plot(test.index, test, label="Actual Values")
plt.plot(test.index, predictions, label="Forecast")
plt.title("ARIMA Forecast vs Actual")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.legend()
plt.show()
