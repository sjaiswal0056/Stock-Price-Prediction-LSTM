
import os
import numpy as np
import yfinance as yf
import joblib

from datetime import datetime, timedelta
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ===================================
# 1. CONFIGURATION
# ===================================

STOCK = "GOOG"
LOOKBACK = 100
EPOCHS = 20
BATCH_SIZE = 32

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Latest_stock_price_model.keras"
SCALER_PATH = BASE_DIR / "stock_scaler.pkl"

print("=" * 50)
print("STOCK PRICE PREDICTION - LSTM TRAINING")
print("=" * 50)

# ===================================
# 2. DOWNLOAD STOCK DATA
# ===================================

print("\nDownloading stock market data...")

end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 20)

try:
    data = yf.download(
        STOCK,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        multi_level_index=False,
        progress=False,
        threads=False,
        timeout=30
    )

    if data.empty:
        raise ValueError("No stock data downloaded.")

except Exception as e:
    print("Error downloading data:", e)
    raise SystemExit(1)

print("Stock data downloaded successfully!")
print("Total records:", len(data))

# ===================================
# 3. DATA PREPROCESSING
# ===================================

print("\nPreprocessing stock data...")

data = data.dropna(subset=["Close"])
close_prices = data[["Close"]].values

if len(close_prices) < 500:
    print("Not enough stock data for training.")
    raise SystemExit(1)

# Split into 70% training and 30% testing
training_size = int(len(close_prices) * 0.70)

training_data = close_prices[:training_size]

# Normalize training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_training_data = scaler.fit_transform(training_data)

# Save scaler
joblib.dump(scaler, SCALER_PATH)

print("Scaler saved successfully!")

# ===================================
# 4. CREATE TRAINING SEQUENCES
# ===================================

X_train = []
y_train = []

for i in range(LOOKBACK, len(scaled_training_data)):
    X_train.append(
        scaled_training_data[i - LOOKBACK:i]
    )
    y_train.append(
        scaled_training_data[i, 0]
    )

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

print("Training data shape:", X_train.shape)

# ===================================
# 5. BUILD LSTM MODEL
# ===================================

print("\nBuilding LSTM model...")

model = Sequential([
    Input(shape=(LOOKBACK, 1)),

    LSTM(50, return_sequences=True),
    Dropout(0.2),

    LSTM(50, return_sequences=True),
    Dropout(0.2),

    LSTM(50),
    Dropout(0.2),

    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

model.summary()

# ===================================
# 6. TRAIN MODEL
# ===================================

print("\nStarting LSTM training...")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)

# ===================================
# 7. SAVE TRAINED MODEL
# ===================================

model.save(MODEL_PATH)

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED!")
print("=" * 50)

print("Saved model:", MODEL_PATH)
print("Saved scaler:", SCALER_PATH)

print("\nYou can now run your Streamlit application.")
