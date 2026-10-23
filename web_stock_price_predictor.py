
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from pathlib import Path
from datetime import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Predictor App")

# Stock input
stock = st.text_input("Enter Stock ID", "GOOG").strip().upper()

if not stock:
    st.warning("Please enter a stock symbol.")
    st.stop()

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Download stock data
try:
    with st.spinner("Downloading stock data..."):
        google_data = yf.download(
            stock,
            start=start,
            end=end,
            auto_adjust=True,
            multi_level_index=False,
            progress=False,
            threads=False,
            timeout=20
        )

    if google_data.empty:
        st.error("No data found. Check your stock symbol or connection.")
        st.stop()

    google_data = google_data.dropna(subset=["Close"])

except Exception as e:
    st.error(f"Stock data download failed: {e}")
    st.stop()

# Load trained model
model_path = Path(__file__).resolve().parent / "Latest_stock_price_model.keras"

if not model_path.exists():
    st.error("Model file Latest_stock_price_model.keras is missing.")
    st.info("Run stock_price.ipynb to train and save the model.")
    st.stop()

try:
    model = load_model(model_path, compile=False)
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()

# Show stock data
st.subheader("Stock Data")
st.dataframe(google_data.tail(100), use_container_width=True)

# Moving averages
google_data["MA_100"] = google_data["Close"].rolling(100).mean()
google_data["MA_200"] = google_data["Close"].rolling(200).mean()
google_data["MA_250"] = google_data["Close"].rolling(250).mean()

def plot_graph(title, columns):
    fig, ax = plt.subplots(figsize=(14, 6))

    for column in columns:
        ax.plot(google_data.index, google_data[column], label=column)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)

st.subheader("Stock Price Moving Averages")

plot_graph("Close Price vs 100-Day MA", ["Close", "MA_100"])
plot_graph("Close Price vs 200-Day MA", ["Close", "MA_200"])
plot_graph("Close Price vs 250-Day MA", ["Close", "MA_250"])
plot_graph(
    "Close Price vs 100-Day and 250-Day MA",
    ["Close", "MA_100", "MA_250"]
)

# Prepare test data
splitting_len = int(len(google_data) * 0.7)

if len(google_data) - splitting_len <= 100:
    st.warning("Not enough stock data to generate predictions.")
    st.stop()

# IMPORTANT:
# For an exact reproduction of training, load the scaler
# fitted on training data. This fallback keeps the original
# project's test-scaling approach.
scaler = MinMaxScaler(feature_range=(0, 1))

x_test = google_data[["Close"]].iloc[splitting_len:]
scaled_data = scaler.fit_transform(x_test)

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data = np.array(x_data)
y_data = np.array(y_data)

# Predict prices
try:
    predictions = model.predict(x_data, verbose=0)

    inv_predictions = scaler.inverse_transform(predictions)
    inv_actual = scaler.inverse_transform(y_data)

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Prediction table
plotting_data = pd.DataFrame({
    "Actual Price": inv_actual.reshape(-1),
    "Predicted Price": inv_predictions.reshape(-1)
}, index=google_data.index[splitting_len + 100:])

st.subheader("Actual vs Predicted Stock Prices")
st.dataframe(plotting_data.tail(100), use_container_width=True)

# Prediction chart
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    plotting_data.index,
    plotting_data["Actual Price"],
    label="Actual Price"
)

ax.plot(
    plotting_data.index,
    plotting_data["Predicted Price"],
    label="Predicted Price"
)

ax.set_title(f"{stock} - Actual vs Predicted Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.success("Stock prediction completed successfully!")
st.caption(
    "Predictions are historical model estimates, not financial advice."
)
