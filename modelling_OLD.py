import pandas as pd
import numpy as np
import xgboost as xgb
import polars as pl
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import sys

target_ins = "NOVO-B.CO"
#target_ins = "CARL-B.CO"
#target_ins = "VWS.CO"
#target_ins = "GMAB.CO"
#target_ins = "DANSKE.CO"
#target_ins = "MAERSK-B.CO"
target_col = "High_"+target_ins
df = pl.read_csv("models/train_data.csv")
df = df.rename({"Datetime":"datetime"})
df = df.rename({target_col:"target"})
df = df.with_columns((pl.col("datetime").str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")).alias("datetime"))
df = df.to_pandas()

# Filter to market hours
def filter_market_hours(df):
    df = df.copy()
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df = df[(df['hour'] >= 9) & ((df['hour'] < 17))]
    return df

def transform_to_daily_relative(df, datetime_col="datetime"):
    """
    Transforms all numeric columns (except datetime_col) into daily relative percentage changes
    based on the last value of each column from the previous day, for data with 5-minute intervals,
    while preserving the original row order.
    
    Parameters:
    - df: pandas DataFrame with a datetime column (5-min intervals) and price-related columns
    - datetime_col: str, name of the datetime column to exclude from transformation (default: "datetime")
    
    Returns:
    - pandas DataFrame with transformed columns (as %-changes relative to previous day's last value)
    """
    # Ensure datetime is in proper format and preserve original order
    df = df.copy()
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Store the original index to preserve order
    original_index = df.index
    
    # Sort by datetime temporarily to ensure correct daily last values
    df = df.sort_values(datetime_col)
    
    # Extract date from datetime for daily grouping
    df['date'] = df[datetime_col].dt.date
    
    # Identify columns to transform (all numeric except datetime_col)
    cols_to_transform = [col for col in df.columns if col != datetime_col and col != 'date']
    
    # Get the last value of each column per day and shift to previous day
    daily_lasts = df.groupby('date')[cols_to_transform].last().shift(1)
    
    # Merge the previous day's last values back into the DataFrame
    df = df.merge(daily_lasts, left_on='date', right_index=True, how='left', suffixes=('', '_prev'))
    
    # Transform each column into %-change relative to its own previous day's last value
    for col in cols_to_transform:
        prev_col = f"{col}_prev"
        df[col]=df[col].astype(float)
        df[prev_col]=df[prev_col].astype(float)
        df[col+"_rel"] = (df[col] - df[prev_col]) / df[prev_col]
    
    # Drop temporary columns (date and all _prev columns)
    df = df.drop(columns=['date'] + [f"{col}_prev" for col in cols_to_transform])
    
    # Restore original order using the saved index
    df = df.loc[original_index].reset_index(drop=True)

    return df

df = filter_market_hours(df)

# Feature Engineering
def create_features(df, target_col='target', horizon=24, max_lag=12, lag_features=None):

    #df = pd.merge(df, transform_to_daily_relative(df), how='inner', on="datetime")
    df = transform_to_daily_relative(df)

    df = df.copy()
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    
    # Future datetime check
    df['future_datetime'] = df['datetime'] + pd.Timedelta(minutes=5 * horizon)
    df['future_date'] = df['future_datetime'].dt.date
    df['future_hour'] = df['future_datetime'].dt.hour
    df['future_minute'] = df['future_datetime'].dt.minute
    
    # Keep only same-day predictions within market hours
    df = df[df['date'] == df['future_date']]
    
    # Set datetime as index
    df = df.set_index('datetime')
    
    # Set target_future
    df['target_future'] = df[target_col].shift(-horizon, freq='5min')
    
    # Prepare all lagged features
    lagged_columns = {}
    lags = [1,5,12]
    for lag in lags:
        lagged_columns[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    if lag_features:
        for feature in lag_features:
            for lag in lags:
                lagged_columns[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    
    # Concatenate all lagged columns
    df = pd.concat([df] + list(lagged_columns.values()), axis=1)
    df.columns = list(df.columns[:-len(lagged_columns)]) + list(lagged_columns.keys())
    
    # Rolling stats
    df_reset = df.reset_index()
    df_reset['target_roll_mean_12'] = df_reset.groupby('date')[target_col].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
    df_reset['target_roll_std_12'] = df_reset.groupby('date')[target_col].rolling(window=12, min_periods=1).std().reset_index(level=0, drop=True)
    df = df_reset.set_index('datetime')
    
    # Time-based features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    
    # Drop rows where target_future is NaN
    df = df.dropna(subset=['target_future'])
    
    # Reset index
    df = df.reset_index()
    df = df[(df['future_hour'] < 17) & (df['future_hour'] > 8)]
    return df

# Prepare data
target_col = 'target'
lag_features = [c for c in df.columns if c not in ["datetime",target_col]]
df = create_features(df, target_col=target_col, lag_features=lag_features)
#df = df.dropna()

# Define features and target
feature_cols = [col for col in df.columns if col not in ['datetime', target_col, 'target_future', 'date', 'time', 'future_datetime', 'future_date', 'future_hour', 'future_minute']]
X = df[feature_cols]
y = df['target_future']

# Train-test split
train_size = int(len(X) * 0.95)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
dates_test = df['datetime'][train_size:]

"""from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
xgb_model = MLPRegressor(
    hidden_layer_sizes=(300, 150, 50),
    max_iter=500,
    learning_rate_init=0.01,
    random_state=42)"""

"""from sklearn.ensemble import RandomForestRegressor
xgb_model = RandomForestRegressor(
    n_estimators=128,
    max_depth=6,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)"""

"""xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=128,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)"""

from sklearn.ensemble import HistGradientBoostingRegressor
xgb_model = HistGradientBoostingRegressor(
    max_iter=256,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Create test DataFrame
test_df = df.iloc[train_size:].copy()
valid_test_dates = test_df['date'].unique()

# Test Single Day
def test_single_day(test_df, model, target_col, test_date, feature_cols, valid_dates):
    test_date = pd.to_datetime(test_date).date()
    if test_date not in valid_dates:
        raise ValueError(f"{test_date} is not in the test set. Valid dates: {valid_test_dates}")
    
    test_day_df = test_df[test_df['date'] == test_date]
    X_test_day = test_day_df[feature_cols]
    y_test_day = test_day_df['target_future']
    dates_test_day = test_day_df['datetime']  # Ensure this is a Pandas Series
    
    y_pred_day = model.predict(X_test_day)
    rmse_day = np.sqrt(mean_squared_error(y_test_day, y_pred_day))
    print(f"RMSE for {test_date}: {rmse_day:.4f}")
    
    # Verify lengths and types
    print(f"Lengths: dates={len(dates_test_day)}, actual={len(y_test_day)}, predicted={len(y_pred_day)}")
    print(f"Type of dates_test_day: {type(dates_test_day)}, dtype: {dates_test_day.dtype}")
    
    assert len(dates_test_day) == len(y_test_day) == len(y_pred_day), \
        f"Length mismatch: dates={len(dates_test_day)}, actual={len(y_test_day)}, predicted={len(y_pred_day)}"
    
    return dates_test_day, y_test_day, y_pred_day

# Plotting Function
def plot_predictions(dates, actual, predicted, title="Actual vs Predicted Stock Prices (Market Hours Only)", xlabel="Time", ylabel="Price"):
    # Ensure dates is a Pandas Series with datetime dtype
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates, name='datetime')
    if not pd.api.types.is_datetime64_any_dtype(dates):
        print(f"Converting dates to datetime. Original dtype: {dates.dtype}")
        dates = pd.to_datetime(dates)
    
    # Verify lengths
    if not (len(dates) == len(actual) == len(predicted)):
        raise ValueError(f"Length mismatch: dates={len(dates)}, actual={len(actual)}, predicted={len(predicted)}")
    
    plot_df = pd.DataFrame({
        'datetime': dates,
        'actual': actual,
        'predicted': predicted
    })
    print(f"plot_df['datetime'] dtype: {plot_df['datetime'].dtype}")
    plot_df['date'] = plot_df['datetime'].dt.date
    plot_df['plot_index'] = range(len(plot_df))
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_df['plot_index'], plot_df['actual'], label='Actual', color='blue', linewidth=1.5)
    plt.plot(plot_df['plot_index'], plot_df['predicted'], label='Predicted', color='orange', linestyle='--', linewidth=1.5)
    
    day_starts = plot_df.groupby('date').head(1).index
    day_labels = plot_df.loc[day_starts, 'datetime'].dt.strftime('%Y-%m-%d 09:00')
    plt.xticks(plot_df.loc[day_starts, 'plot_index'], day_labels, rotation=45)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Test and plot
test_date = valid_test_dates[0]
dates_test_day, y_test_day, y_pred_day = test_single_day(test_df, xgb_model, target_col, test_date, feature_cols, valid_test_dates)
#plot_predictions(dates_test_day, y_test_day, y_pred_day, title=f"Actual vs Predicted Stock Prices ({test_date})")
#sys.exit()

# Probability Function
def predict_price_increase_probability(model, X_latest, current_price, percentage_increase, horizon=24, n_simulations=1000):
    if isinstance(X_latest, pd.DataFrame):
        X_latest = X_latest.values
    pred = model.predict(X_latest)[0]
    residuals = y_test_day - y_pred_day
    residual_std = np.std(residuals)
    simulated_prices = np.random.normal(loc=pred, scale=residual_std, size=n_simulations)
    target_price = current_price * (1 + percentage_increase / 100)
    prob = np.mean(simulated_prices > target_price)
    return prob

# Example probability
"""X_latest = X_test.tail(1)
current_price = test_df[target_col].iloc[-25]
percentage_increase = 1
prob = predict_price_increase_probability(xgb_model, X_latest, current_price, percentage_increase)
print(f"Probability of {percentage_increase}% increase in 2 hours on {test_date}: {prob:.2%}")
"""
# Optional: Plot all test days
plot_predictions(test_df['datetime'], test_df['target_future'], xgb_model.predict(X_test), title="Actual vs Predicted (All Test Days Collapsed)")
