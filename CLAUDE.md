# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a day trading application called "day2day" that helps with day trading activities in the stock market. The application is designed to:

1. Collect market data from Saxo Bank API
2. Create and evaluate time series models
3. Use backtesting to test trading strategies
4. Predict stock prices 2 hours into the future

## Installation and Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

The project uses setuptools for installation:
```bash
python setup.py install
```

## Key Dependencies

- **pandas**: Data manipulation and analysis
- **polars**: Preferred for CSV operations to minimize load time
- **xgboost**: Machine learning models
- **scikit-learn**: Additional ML algorithms
- **requests**: API calls to Saxo Bank
- **matplotlib/seaborn**: Data visualization
- **numpy**: Numerical computations
- **yfinance**: Alternative market data source

## Project Structure

- `auth.py`: OAuth authentication with Saxo Bank API
- `market.py`: Market data collection and processing
- `data_prep.py`: Data preparation and feature engineering
- `modelling.py`: ML model creation and training
- `sim.py`: Backtesting and trading strategy simulation
- `about.py`: Project metadata

## Common Commands

### Data Collection
```bash
python market.py
```
Collects Danish stock data from Saxo Bank API with 1-minute granularity.

### Data Preparation
```bash
python data_prep.py
```
Prepares training data with features and creates `models/train_data.csv`.

### Model Training
```bash
python modelling.py
```
Trains XGBoost models for price prediction.

### Backtesting
```bash
python sim.py
```
Runs backtesting strategies on collected data.

## Architecture Notes

### Data Collection (market.py)
- Uses Saxo Bank API with 0.95-second delays to avoid rate limits
- Collects data in 3-day intervals due to API limitations
- Stores data in CSV format using Polars for performance
- Supports seamless updates without duplicating datetime-instrument pairs

### Data Preparation (data_prep.py)
- Transforms raw market data into ML features
- Supports both raw prices and percentage changes
- Creates lagged features with configurable time horizons
- Fills missing datetime values using forward fill strategy
- Filters for high-quality data (>25 observations per day)

### Modeling (modelling.py)
- Uses XGBoost for price prediction
- Transforms data to daily relative percentage changes
- Creates features with multiple lag periods (1, 5, 12 intervals)
- Supports 2-hour prediction horizon (24 5-minute intervals)
- Includes rolling statistics and time-based features

### Backtesting (sim.py)
- Implements take-profit and stop-loss strategies
- Supports fixed investment amounts with $20 trade costs
- Processes data day-by-day with position carryover
- Includes comprehensive trade statistics and performance metrics

## API Configuration

The Saxo Bank API requires:
- Client ID and Client Secret (stored in auth.py)
- OAuth2 flow for authentication
- Access tokens for API calls

**Security Note**: API credentials are currently hardcoded in `auth.py` and should be moved to environment variables or a separate config file not committed to git.

## Data Storage

- Raw market data: `ins_data/` directory
- Training data: `models/train_data.csv`
- Backtesting results: `backtesting_data/` directory

## Target Instrument

The primary target for modeling is "NOVO-B.CO" (Novo Nordisk B shares), but the system supports multiple Danish stocks from Copenhagen Stock Exchange.

## Trading Strategy

The backtesting system implements:
- Long-only day trading strategy
- Positions must be closed by end of day
- Configurable take-profit and stop-loss levels
- Minimum price change triggers for entry
- Fixed investment amounts per trade