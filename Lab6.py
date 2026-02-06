import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

df = pd.read_csv("gios-pjp-data.csv", sep=';', decimal=',')
df = df[["date", "PM10"]]
df = df.dropna()

values = df["PM10"].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

look_back = 30
X, y = [], []

for i in range(look_back, len(scaled)):
    X.append(scaled[i - look_back:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model_lstm = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])

model_lstm.compile(optimizer="adam", loss="mse")
model_lstm.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

model_gru = Sequential([
    GRU(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    GRU(50),
    Dense(1)
])

model_gru.compile(optimizer="adam", loss="mse")
model_gru.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

pred_lstm = model_lstm.predict(X_test)
pred_gru = model_gru.predict(X_test)

pred_lstm = scaler.inverse_transform(pred_lstm)
pred_gru = scaler.inverse_transform(pred_gru)
real = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(real, label="Rzeczywiste PM10")
plt.plot(pred_lstm, label="LSTM")
plt.plot(pred_gru, label="GRU")
plt.legend()
plt.grid(True)
plt.show()