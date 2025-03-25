import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import requests
from scipy.stats import norm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# User inputs
symbol = st.text_input("Enter Stock Symbol:", value="AAPL")
strike_price = st.number_input("Enter Strike Price:", value=100.0)
time_to_expiration = st.number_input("Time to Expiration (in years):", value=0.5)
risk_free_rate = st.number_input("Risk-Free Rate (in %):", value=2.0)

# Fetch stock data from Polygon.io
def fetch_stock_data(symbol, api_key):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/30/2024?apiKey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            df = pd.DataFrame(data["results"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")  # Convert timestamp to datetime
            df.rename(columns={"c": "close"}, inplace=True)
            return df
    return None

stock_data = fetch_stock_data(symbol, POLYGON_API_KEY)

if stock_data is None or stock_data.empty:
    st.error("Error: Could not fetch stock data. Check the stock symbol or API key.")
    st.stop()

# Ensure stock_data has required columns
if 'close' not in stock_data or 't' not in stock_data:
    st.error("Error: Stock data is missing required columns ('close' for price and 't' for timestamp).")
    st.stop()

# Calculate Volatility
stock_data["return"] = stock_data["close"].pct_change()
if len(stock_data) < 20:
    stock_data["volatility"] = stock_data["return"].std()
else:
    stock_data["volatility"] = stock_data["return"].rolling(window=20).std()

# Annualized volatility calculation
annual_volatility = stock_data["volatility"].iloc[-1] * np.sqrt(252)
st.session_state["volatility"] = annual_volatility if not np.isnan(annual_volatility) else 0.0

# Fill NaN values to prevent issues
stock_data["volatility"].fillna(0, inplace=True)

# Black-Scholes function
def black_scholes(call_put, S, K, T, r, sigma):
    if sigma <= 0:  # Prevent divide-by-zero errors
        return 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if call_put == 'call':
        option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif call_put == 'put':
        option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return option_price

# Calculate Black-Scholes call & put prices
S = stock_data["close"].iloc[-1]  # Latest stock price
call_price = black_scholes('call', S, strike_price, time_to_expiration, risk_free_rate / 100, st.session_state["volatility"])
put_price = black_scholes('put', S, strike_price, time_to_expiration, risk_free_rate / 100, st.session_state["volatility"])

# Display option prices
st.markdown(f"### **Call Option Price: ${call_price:.2f}**")
st.markdown(f"### **Put Option Price: ${put_price:.2f}**")

# Visualization - Stock Price and Volatility
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.set_facecolor("#f4f4f4")

# Stock Price Plot
ax1.set_xlabel("Date", fontsize=12, fontweight='bold')
ax1.set_ylabel("Stock Price (USD)", color='blue', fontsize=12, fontweight='bold')
ax1.plot(stock_data["t"], stock_data["close"], color='blue', linewidth=2, marker='o', markersize=5, label='Stock Price')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, linestyle='--', alpha=0.6)

# Volatility Plot on Secondary Axis with Gradient Fill
ax2 = ax1.twinx()
ax2.set_ylabel("Volatility (%)", color='red', fontsize=12, fontweight='bold')
ax2.plot(stock_data["t"], stock_data["volatility"] * 100, color='red', linestyle='dashed', linewidth=2, label='Volatility')
ax2.fill_between(stock_data["t"], stock_data["volatility"] * 100, color='red', alpha=0.3)
ax2.tick_params(axis='y', labelcolor='red')

# Enhanced Background and Legend
fig.tight_layout()
fig.patch.set_facecolor('#eaeaea')
ax1.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor='black')
ax2.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='black')

# Display the final graph
st.pyplot(fig)
