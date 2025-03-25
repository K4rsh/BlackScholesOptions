# Black-Scholes Options and Earnings Surprise Impact Analyzer

This project provides two Streamlit applications for analyzing stock data and options pricing using the Black-Scholes model. It leverages the Polygon.io API for fetching historical stock data and calculates key metrics like volatility and option prices.

## Features

### App 1: Earnings Surprise Impact Analyzer (`app.py`)
- Fetches historical stock data for a given symbol and date range.
- Calculates daily returns and rolling volatility.
- Computes annualized volatility.
- Provides an interactive interface for calculating Black-Scholes option prices (Call and Put).
- Visualizes stock prices and volatility with enhanced graphs.

### App 2: Black-Scholes Options Pricing (`app2.py`)
- Fetches stock data for a specific symbol using Polygon.io.
- Calculates historical volatility and annualized volatility.
- Implements the Black-Scholes model to compute Call and Put option prices.
- Visualizes stock prices and volatility trends.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/BlackScholesOptions.git
   cd BlackScholesOptions