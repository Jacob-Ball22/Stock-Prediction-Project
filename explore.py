import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_close_price_over_time(df, title="S&P 500 Closing Price Over Time"):
    """
    Plot the closing price over time.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], linewidth=1)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Closing Price ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_target_distribution(df):
    """
    Plot the distribution of the target variable.
    """
    target_counts = df['Target'].value_counts()
    target_pct = df['Target'].value_counts(normalize=True) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    #Count plot
    ax1.bar(['Down (0)', 'Up (1)'], target_counts.values, color=['#d62728', '#2ca02c'])
    ax1.set_title('Target Variable Distribution (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    #Percentage plot
    ax2.bar(['Down (0)', 'Up (1)'], target_pct.values, color=['#d62728', '#2ca02c'])
    ax2.set_title('Target Variable Distribution (Percentage)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    print(f"Days price went up: {target_counts[1]} ({target_pct[1]:.2f}%)")
    print(f"Days price went down: {target_counts[0]} ({target_pct[0]:.2f}%)")

def plot_rolling_averages(df, horizons=[5, 60, 250]):
    """
    Plot closing price with rolling averages.
    """
    plt.figure(figsize=(14, 7))
    
    #Plot actual closing price
    plt.plot(df.index, df['Close'], label='Actual Close', linewidth=1.5, alpha=0.7)
    
    #Plot rolling averages
    colors = ['orange', 'red', 'purple']
    for horizon, color in zip(horizons, colors):
        rolling_avg = df['Close'].rolling(horizon).mean()
        plt.plot(df.index, rolling_avg, label=f'{horizon}-Day MA', 
                linewidth=1.5, alpha=0.7, color=color)
    
    plt.title('S&P 500 Closing Price with Moving Averages', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_volume_over_time(df):
    """
    Plot trading volume over time.
    """
    plt.figure(figsize=(14, 6))
    plt.bar(df.index, df['Volume'], width=1, color='steelblue', alpha=0.7)
    plt.title('S&P 500 Trading Volume Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volume', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_heatmap(df, features=None):
    """
    Plot correlation heatmap of features.
    """
    if features is None:
        #Use all numeric columns except Tomorrow
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Tomorrow' in features:
            features.remove('Tomorrow')
    corr_matrix = df[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_price_volatility(df, window=30):
    """
    Plot rolling standard deviation (volatility) of closing prices.
    """
    volatility = df['Close'].rolling(window).std()
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, volatility, color='darkred', linewidth=1)
    plt.title(f'S&P 500 {window}-Day Rolling Volatility (Std Dev)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_returns_distribution(df):
    """
    Plot the distribution of daily returns.
    """
    #Calculate daily returns
    returns = df['Close'].pct_change() * 100
    plt.figure(figsize=(14, 6))
    
    #Histogram
    plt.hist(returns.dropna(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
    plt.xlabel('Daily Return (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    print(f"Mean daily return: {returns.mean():.4f}%")
    print(f"Std dev of returns: {returns.std():.4f}%")
    print(f"Min return: {returns.min():.2f}%")
    print(f"Max return: {returns.max():.2f}%")

def show_basic_statistics(df):
    """
    Display basic statistics about the dataset.
    """
    print("Basic Statistics about the S&P500 Dataset")
    print(f"Total rows: {len(df):,}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Number of years: {(df.index.max() - df.index.min()).days / 365.25:.1f}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
