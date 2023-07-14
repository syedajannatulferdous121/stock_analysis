import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_stock_data(symbol, start_date, end_date):
    """
    Retrieve historical stock data using yfinance.
    """
    stock = yf.download(symbol, start=start_date, end=end_date)
    return stock

def calculate_returns(stock_data):
    """
    Calculate daily returns of the stock.
    """
    close_prices = stock_data['Adj Close']
    returns = close_prices.pct_change()
    return returns

def calculate_moving_averages(stock_data, window_sizes=[50, 200]):
    """
    Calculate moving averages for different window sizes.
    """
    moving_averages = {}
    for window in window_sizes:
        ma = stock_data['Adj Close'].rolling(window).mean()
        moving_averages[f'{window}MA'] = ma
    return moving_averages

def calculate_rsi(stock_data, window=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    close_prices = stock_data['Adj Close']
    delta = close_prices.diff()
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    average_gain = up.rolling(window).mean()
    average_loss = down.rolling(window).mean()
    rs = average_gain / average_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_stock_data(stock_data, moving_averages, rsi_data):
    """
    Plot the stock's adjusted close prices, moving averages, and RSI.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    
    # Plot stock prices
    ax1.plot(stock_data.index, stock_data['Adj Close'], label='Stock Price')
    ax1.set_ylabel('Price')

    # Plot moving averages
    for ma_label, ma_values in moving_averages.items():
        ax1.plot(stock_data.index, ma_values, label=ma_label)
    
    # Plot RSI
    ax2.plot(stock_data.index, rsi_data, color='orange')
    ax2.axhline(70, color='red', linestyle='--')
    ax2.axhline(30, color='green', linestyle='--')
    ax2.set_ylabel('RSI')
    
    # Plot volume
    ax3.bar(stock_data.index, stock_data['Volume'], label='Volume', alpha=0.3)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Volume')

    # Set title and legend
    plt.suptitle('Stock Analysis')
    ax1.legend()
    
    plt.show()

def train_linear_regression_model(stock_data):
    """
    Train a linear regression model to predict future stock prices.
    """
    features = stock_data[['Open', 'High', 'Low', 'Volume']]
    target = stock_data['Close']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    print(f'Train RMSE: {train_rmse:.2f}')
    print(f'Test RMSE: {test_rmse:.2f}')

def run_stock_analysis():
    """
    Run the stock analysis based on user inputs.
    """
    # Get user inputs
    symbol = input("Enter the stock symbol (e.g., AAPL): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    
    # Retrieve stock data
    stock_data = get_stock_data(symbol, start_date, end_date)
    if stock_data.empty:
        print("No data available for the specified symbol and date range.")
        return
    
    returns = calculate_returns(stock_data)
    moving_averages = calculate_moving_averages(stock_data)
    rsi_data = calculate_rsi(stock_data)
    plot_stock_data(stock_data, moving_averages, rsi_data)
    train_linear_regression_model(stock_data)

# Run the stock analysis based on user inputs
run_stock_analysis()
