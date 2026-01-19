# =========================================================
# Solid Waste Forecasting using ML & DL Models
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, GRU, SimpleRNN, Dense, Dropout, Bidirectional
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

from statsmodels.tsa.statespace.sarimax import SARIMAX

# =========================================================
# 1. LOAD DATA (UPDATE PATHS LOCALLY)
# =========================================================

df1 = pd.read_excel("dataset_1_2021.xlsx")
df2 = pd.read_excel("dataset_2_2022.xlsx")
df3 = pd.read_excel("dataset_3_2023.xlsx")
df4 = pd.read_excel("dataset_4_2024.xlsx")

df = pd.concat([df1, df2, df3, df4], ignore_index=True)

df["Date"] = pd.to_datetime(df["Date"])
df = df[["Date", "Total Waste"]]
df = df.sort_values("Date").reset_index(drop=True)

# =========================================================
# 2. TRAIN–TEST SPLIT (TIME-BASED)
# =========================================================

train_df = df[(df["Date"].dt.year < 2024) |
              ((df["Date"].dt.year == 2024) & (df["Date"].dt.month < 9))]

test_df = df[(df["Date"].dt.year == 2024) &
             (df["Date"].dt.month >= 9)]

# =========================================================
# 3. SCALING
# =========================================================

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[["Total Waste"]])
test_scaled = scaler.transform(test_df[["Total Waste"]])

# =========================================================
# 4. SEQUENCE CREATION
# =========================================================

def create_dataset(data, time_steps=30):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

TIME_STEPS = 30

X_train, y_train = create_dataset(train_scaled, TIME_STEPS)
X_test, y_test = create_dataset(test_scaled, TIME_STEPS)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# =========================================================
# 5. LSTM MODEL
# =========================================================

lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(TIME_STEPS, 1)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(1)
])

lstm_model.compile(
    optimizer=Adam(0.001),
    loss="mean_squared_error"
)

lstm_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[ReduceLROnPlateau(patience=3)]
)

# =========================================================
# 6. GRU MODEL
# =========================================================

gru_model = Sequential([
    Bidirectional(GRU(128, return_sequences=True), input_shape=(TIME_STEPS, 1)),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(1)
])

gru_model.compile(
    optimizer=Adam(0.001),
    loss="mean_squared_error"
)

gru_model.fit(X_train, y_train, epochs=30, batch_size=32)

# =========================================================
# 7. RNN MODEL
# =========================================================

rnn_model = Sequential([
    SimpleRNN(64, input_shape=(TIME_STEPS, 1)),
    Dropout(0.2),
    Dense(1)
])

rnn_model.compile(
    optimizer=Adam(0.001),
    loss="mean_squared_error"
)

rnn_model.fit(X_train, y_train, epochs=30, batch_size=32)

# =========================================================
# 8. LINEAR REGRESSION
# =========================================================

X_lr_train = train_scaled[:-1]
y_lr_train = train_scaled[1:]

X_lr_test = test_scaled[:-1]
y_lr_test = test_scaled[1:]

lr_model = LinearRegression()
lr_model.fit(X_lr_train, y_lr_train)

# =========================================================
# 9. SARIMA MODEL
# =========================================================

sarima_train = train_df.set_index("Date")["Total Waste"]

sarima_model = SARIMAX(
    sarima_train,
    order=(5, 1, 0),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(disp=False)

sarima_pred = sarima_fit.forecast(steps=len(test_df))

# =========================================================
# 10. EVALUATION FUNCTION
# =========================================================

def evaluate_model(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# =========================================================
# 11. EVALUATE MODELS
# =========================================================

lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test))
gru_pred = scaler.inverse_transform(gru_model.predict(X_test))
rnn_pred = scaler.inverse_transform(rnn_model.predict(X_test))

y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

evaluate_model(y_true, lstm_pred, "LSTM")
evaluate_model(y_true, gru_pred, "GRU")
evaluate_model(y_true, rnn_pred, "RNN")
evaluate_model(
    scaler.inverse_transform(y_lr_test.reshape(-1, 1)),
    scaler.inverse_transform(lr_model.predict(X_lr_test).reshape(-1, 1)),
    "Linear Regression"
)
evaluate_model(test_df["Total Waste"].values[:len(sarima_pred)], sarima_pred, "SARIMA")

# =========================================================
# 12. PLOT RESULTS (OPTIONAL)
# =========================================================

plt.figure(figsize=(12, 6))
plt.plot(test_df["Date"].iloc[TIME_STEPS:], y_true, label="Actual")
plt.plot(test_df["Date"].iloc[TIME_STEPS:], lstm_pred, label="LSTM")
plt.plot(test_df["Date"].iloc[TIME_STEPS:], gru_pred, label="GRU")
plt.plot(test_df["Date"].iloc[TIME_STEPS:], rnn_pred, label="RNN")
plt.legend()
plt.title("Solid Waste Forecasting Comparison")
plt.show()
