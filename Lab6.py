import numpy as np, pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Download stock market data
data = yf.download('AAPL', start='2023-01-01', end='2024-01-01', auto_adjust=False)
prices = data['Close'].values.reshape(-1,1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(prices)

# Prepare dataset
def create_data(s, steps=20):
    X, y = [], []
    for i in range(len(s)-steps):
        X.append(s[i:i+steps,0])
        y.append(s[i+steps,0])
    return np.array(X), np.array(y)

steps = 20
X, y = create_data(scaled, steps)
X = X.reshape((-1, steps, 1))

# Build LSTM (new recommended style)
model = Sequential([
    Input(shape=(steps,1)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, batch_size=8, verbose=0)

# Predict
pred_scaled = model.predict(X)
pred = scaler.inverse_transform(pred_scaled)

plt.figure(figsize=(8,4))
plt.plot(prices[steps:], label='True')
plt.plot(pred, label='Predicted')
plt.title('AAPL Price Forecast')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()
