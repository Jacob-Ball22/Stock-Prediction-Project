# S&P 500 Stock Market Prediction Project

## Project Overview
This project uses machine learning to predict whether the S&P 500 stock index will increase on the following trading day. Using nearly 100 years of historical data and Random Forest classification, the model achieves 55.7% precision in predicting upward price movements.

## Models Tested
1. **Random Forest (100 trees)** — Baseline model
2. **Random Forest (200 trees)** — Improved model with more estimators
3. **Random Forest with Threshold** — Uses 60% probability threshold for predictions

## Results
- **Best Model**: Random Forest with 200 trees
- **Precision**: 55.7%
- **Baseline**: 54.6% (always predicting "up")

## Key Findings
- Short-term momentum (2–5 day trends) are most predictive
- Longer-term moving averages provide important context
- Market volatility spikes during crisis periods (2008, 2020)

## Technologies Used
- **Python 3.11**
- **Libraries**: yfinance, pandas, numpy, matplotlib, seaborn, scikit-learn
- **ML Algorithm**: Random Forest Classifier
- **Validation**: Walk-forward backtesting

## Future Improvements
- Incorporate external market indicators (VIX, economic data)
- Test additional algorithms (XGBoost, LSTM, GRU)
- Implement more sophisticated technical indicators (RSI, MACD, Bollinger Bands)

## Author
Jacob Ball — Master's in Data Science, Fordham University
