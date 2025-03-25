import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Streamlit Title
st.title("Earnings Surprise Impact Analyzer")

def get_historical_stock_data(symbol, start_date, end_date):
    """Fetch historical stock data for a range of dates using Polygon.io API."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}?apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url)
    data = response.json()
    
    if "status" in data and data["status"] == "OK":
        return pd.DataFrame(data['results'])
    else:
        st.error(f"Error fetching data: {data}")  # Show error in UI
        return None

# User Input for Stock Symbol and Date Range
stock_symbol = st.text_input("Enter Stock Symbol:", "AAPL")
start_date = st.date_input("Select Start Date:", pd.to_datetime("2024-01-01"))
end_date = st.date_input("Select End Date:", pd.to_datetime("2024-02-23"))

# Fetch Stock Data
if st.button("Get Stock Data"):
    stock_data = get_historical_stock_data(stock_symbol, str(start_date), str(end_date))

    if stock_data is not None:
        stock_data['return'] = stock_data['c'].pct_change()
        
        if len(stock_data) >= 20:
            stock_data['volatility'] = stock_data['return'].rolling(window=20).std()
        else:
            stock_data['volatility'] = stock_data['return'].std()
        
        annual_volatility = stock_data['volatility'].iloc[-1] * math.sqrt(252)
        st.session_state['stock_data'] = stock_data
        st.session_state['volatility'] = annual_volatility
        
        # Display volatility using Streamlit metric
        st.metric("Calculated Volatility", f"{annual_volatility * 100:.2f}%")

# Retrieve stored data from session state
if 'stock_data' in st.session_state:
    stock_data = st.session_state['stock_data']
    st.write("### Stock Data")
    st.dataframe(stock_data)
    
    # Sliders for interactive Black-Scholes inputs
    strike_price = st.slider("Strike Price", min_value=stock_data['c'].min(), max_value=stock_data['c'].max(), value=stock_data['c'].iloc[-1])
    risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=3.0)
    time_to_expiration = st.slider("Time to Expiration (Years)", min_value=0.01, max_value=2.0, value=0.5)
    
    # Black-Scholes function
    def black_scholes(call_put, S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        if call_put == 'call':
            option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        elif call_put == 'put':
            option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return option_price
    
    call_price = black_scholes('call', stock_data['c'].iloc[-1], strike_price, time_to_expiration, risk_free_rate / 100, st.session_state['volatility'])
    put_price = black_scholes('put', stock_data['c'].iloc[-1], strike_price, time_to_expiration, risk_free_rate / 100, st.session_state['volatility'])
    
    st.markdown(f"### **Call Option Price: ${call_price:.2f}**")
    st.markdown(f"### **Put Option Price: ${put_price:.2f}**")
    
    # Visualization - Enhanced Graph with Gradient Fill for Volatility
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_facecolor("#f4f4f4")

    # Stock Price Plot
    ax1.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Stock Price (USD)", color='blue', fontsize=12, fontweight='bold')
    ax1.plot(stock_data['t'], stock_data['c'], color='blue', linewidth=2, marker='o', markersize=5, label='Stock Price')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Volatility Plot on Secondary Axis with Gradient Fill
    ax2 = ax1.twinx()
    ax2.set_ylabel("Volatility (%)", color='red', fontsize=12, fontweight='bold')
    ax2.plot(stock_data['t'], stock_data['volatility'] * 100, color='red', linestyle='dashed', linewidth=2, label='Volatility')
    ax2.fill_between(stock_data['t'], stock_data['volatility'] * 100, color='red', alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Enhanced Background and Legend
    fig.tight_layout()
    fig.patch.set_facecolor('#eaeaea')
    ax1.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor='black')
    ax2.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', edgecolor='black')
    
    # Display the final graph
    st.pyplot(fig)
