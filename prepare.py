import pandas as pd
import numpy as np

def clean_sp500_data(df):
    """
    Clean S&P 500 data by removing timezone info and normalizing dates.
    """
    df.index = df.index.tz_localize(None).normalize()
    return df

def remove_unnecessary_columns(df):
    """
    Remove columns that aren't needed for modeling.
    """
    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    available_columns = []
    for col in columns_to_keep:
        if col in df.columns:
            available_columns.append(col)
    return df[available_columns]

def create_target_variable(df, horizon=1):
    """
    Create target variable indicating if price will go up tomorrow.
    """
    df = df.copy()
    df['Tomorrow'] = df['Close'].shift(-horizon)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    return df

def create_rolling_features(df, horizons=[2, 5, 60, 250, 1000]):
    """
    Create rolling average features for different time horizons.
    """
    df = df.copy()
    for horizon in horizons:
        rolling_avg = df['Close'].rolling(horizon).mean()
        df[f'Close_Ratio_{horizon}'] = df['Close'] / rolling_avg
    return df

def create_trend_features(df, horizons=[2, 5, 60, 250, 1000]):
    """
    Create trend features based on rolling sums of target variable.
    """
    df = df.copy()
    for horizon in horizons:
        df[f'Trend_{horizon}'] = df['Target'].shift(1).rolling(horizon).sum()
    return df

def prepare_sp500_for_modeling(df, start_date="1990-01-01"):
    """
    Complete preparation pipeline for S&P 500 data.
    """
    df = clean_sp500_data(df)
    df = remove_unnecessary_columns(df)
    df = create_target_variable(df)
    df = df.loc[start_date:].copy()
    
    #Create rolling features
    df["Close_Ratio_2"] = df["Close"] / df["Close"].shift(1)
    df["Close_Ratio_5"] = df["Close"] / df["Close"].shift(5)
    df["Close_Ratio_60"] = df["Close"] / df["Close"].shift(60)
    df["Close_Ratio_250"] = df["Close"] / df["Close"].shift(250)
    df["Close_Ratio_1000"] = df["Close"] / df["Close"].shift(1000)
    
    #Create trend features
    df["Trend_2"] = df["Close"].shift(1).rolling(2).sum()
    df["Trend_5"] = df["Close"].shift(1).rolling(5).sum()
    df["Trend_60"] = df["Close"].shift(1).rolling(60).sum()
    df["Trend_250"] = df["Close"].shift(1).rolling(250).sum()
    df["Trend_1000"] = df["Close"].shift(1).rolling(1000).sum()
    
    #Drop any rows with NaN values
    df = df.dropna()
    return df

def split_train_test(df, train_start_row=2500):
    """
    Split data into training and testing sets based on time.
    """
    train = df.iloc[:train_start_row].copy()
    test = df.iloc[train_start_row:].copy()
    return train, test
