import pandas as pd
import numpy as np
import polars as pl
import sys
from datetime import datetime

def str_to_dt(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")

def prepare_train_data():

    mdf = pl.concat([pl.read_csv("ins_data/cop_nasd_5m.csv"),pl.read_csv("ins_data/forex.csv")])
    mdf = mdf.with_columns((pl.col("Datetime").str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")).alias("Datetime"))
    mdf = get_high_quality_data(mdf)
    tickers = list(mdf["ticker"].unique())
    mdf = pl.concat(fill_missing_datetime(mdf.filter(pl.col("ticker")=="NOVO-B.CO"),[mdf.filter(pl.col("ticker")==tick) for tick in tickers]))

    new_df = mdf.filter(pl.col("ticker")==tickers[0]).select(["High","Low","Open","Close","Datetime"]).rename({"High":"High_{}".format(tickers[0]),"Low":"Low_{}".format(tickers[0]),"Open":"Open_{}".format(tickers[0]),"Close":"Close_{}".format(tickers[0])}) 
    for tick in tickers:
        if  tick != tickers[0]:
            ndf = mdf.filter(pl.col("ticker")==tick).select(["High","Low","Open","Close","Datetime"]).rename({"High":"High_{}".format(tick),"Low":"Low_{}".format(tick),"Open":"Open_{}".format(tick),"Close":"Close_{}".format(tick)})
            new_df = new_df.join(ndf,on="Datetime",how="left")
    new_df.write_csv("models/train_data.csv")

def fill_missing_datetime(reference_df, dataframes=[]):
    """
    Fill missing datetime values in multiple dataframes based on reference_df datetimes,
    using each dataframe's own previous values for non-datetime columns, without a fixed interval.
    
    Parameters:
    reference_df: Polars DataFrame providing the reference datetime values
    *dataframes: Variable number of Polars DataFrames to be filled
    
    Returns:
    List of filled Polars DataFrames in the same order as input
    """
    # Convert reference_df to Polars if it isn't already and sort
    if not isinstance(reference_df, pl.DataFrame):
        reference_df = pl.DataFrame(reference_df)
    reference_df = (reference_df
                   .with_columns(pl.col('Datetime').cast(pl.Datetime))
                   .sort('Datetime'))
    
    # Collect all unique datetime values from all dataframes
    all_dfs = [reference_df] + list(dataframes)
    all_dates = (pl.concat([df.select('Datetime') for df in all_dfs])
                .unique()
                .sort('Datetime'))
    
    # Process each input dataframe
    result_dfs = []
    for df in dataframes:
        # Convert to Polars if needed, ensure datetime format and sort
        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
        df = (df
             .with_columns(pl.col('Datetime').cast(pl.Datetime))
             .sort('Datetime'))
        
        # Join with all unique dates and forward fill
        filled_df = (all_dates
                    .join(df, on='Datetime', how='left')
                    .fill_null(strategy='forward'))
        
        result_dfs.append(filled_df)
    
    return result_dfs

def get_high_quality_data(df):

    # Extract date from datetime
    df = df.with_columns(
        pl.col("Datetime").dt.date().alias("Date")
    )
    
    # Group by ticker and date, count rows
    daily_counts = (df
        .groupby(["ticker", "Date"])
        .agg(pl.count().alias("count"))
    )
    
    # Group by ticker and calculate mean of daily counts
    mean_counts = (daily_counts
        .groupby("ticker")
        .agg(pl.col("count").mean())
    )
    mean_counts.sort("count",descending=True).write_csv("mean_counts.csv")
    df = df.filter(pl.col("ticker").is_in(mean_counts.filter(pl.col("count")>75.0)["ticker"].to_list()))
    return df


def backtest_trading_strategy(df, take_profit_pct, stop_loss_pct, low_change_pct, investment_amount):
    """
    Backtest a trading strategy with take profit, stop loss, low price change trigger, 
    fixed investment, positions carried over across days, extended metrics, $20 trade cost,
    and conditional low_change_pct behavior.
    Args:
        df: DataFrame with 'High' and 'Low' prices at 5-minute intervals
        take_profit_pct: Percentage gain target (e.g., 0.02 for 2%)
        stop_loss_pct: Percentage loss limit (e.g., 0.01 for 1%)
        low_change_pct: Minimum percentage change in Low price to trigger buy 
                        (positive: absolute change, negative: upward change only, e.g., -0.005 for 0.5% up)
        investment_amount: Fixed dollar amount to invest per trade
    Returns:
        dict: Results including total profit, trades, and daily/weekly statistics
    """
    # Convert percentages to decimals
    take_profit = 1 + take_profit_pct
    stop_loss = 1 - stop_loss_pct
    min_low_change = abs(low_change_pct)  # Use absolute value for threshold
    TRADE_COST = 20  # Fixed cost per trade in dollars
    
    # Initialize variables
    total_profit = 0
    trades = []
    in_position = False
    entry_price = 0
    position_size = 0
    entry_time = None
    
    # Calculate percentage change in Low prices for the entire dataframe
    df['low_pct_change'] = df['Low'].pct_change()
    
    # Group by date to process one day at a time
    for date, day_df in df.groupby(df.index.date):
        # Get trading window: 1 hour after open to 5 min before close
        day_start = pd.Timestamp(f'{date} 10:30:00')  # Assuming 9:30 ET open + 1 hour
        day_end = pd.Timestamp(f'{date} 15:55:00')    # 4:00 ET close - 5 minutes
        
        # Filter for trading hours
        day_df = day_df[(day_df.index >= day_start) & (day_df.index <= day_end)]
        
        if day_df.empty:
            continue
            
        # Process each 5-minute interval
        for timestamp, row in day_df.iterrows():
            current_high = row['High']
            current_low = row['Low']
            low_pct_change = row['low_pct_change']
            
            if not in_position:
                # Check if Low price change meets threshold
                if not df['low_pct_change'].isna().loc[timestamp]:
                    # If low_change_pct is positive, use absolute change (up or down)
                    if low_change_pct >= 0 and abs(low_pct_change) >= min_low_change:
                        entry_price = current_low
                        position_size = investment_amount / entry_price
                        in_position = True
                        entry_time = timestamp
                    # If low_change_pct is negative, only trigger on upward move
                    elif low_change_pct < 0 and low_pct_change >= min_low_change:
                        entry_price = current_low
                        position_size = investment_amount / entry_price
                        in_position = True
                        entry_time = timestamp
                    
            elif in_position:
                # Check if we hit take profit or stop loss
                take_profit_price = entry_price * take_profit
                stop_loss_price = entry_price * stop_loss
                
                if current_high >= take_profit_price:
                    # Take profit triggered
                    exit_price = take_profit_price
                    raw_profit = position_size * (exit_price - entry_price)
                    profit = raw_profit - TRADE_COST
                    total_profit += profit
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': position_size,
                        'profit': profit
                    })
                    in_position = False
                    
                elif current_low <= stop_loss_price:
                    # Stop loss triggered
                    exit_price = stop_loss_price
                    raw_profit = position_size * (exit_price - entry_price)
                    profit = raw_profit - TRADE_COST
                    total_profit += profit
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': position_size,
                        'profit': profit
                    })
                    in_position = False
    
    # Calculate statistics
    num_trades = len(trades)
    winning_trades = len([t for t in trades if t['profit'] > 0])
    win_rate = winning_trades / num_trades if num_trades > 0 else 0
    
    # Convert trades to DataFrame for easier analysis
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df['exit_date'] = trades_df['exit_time'].dt.date
        trades_df['week'] = trades_df['exit_time'].dt.isocalendar().week
        
        # Daily statistics
        daily_profits = trades_df.groupby('exit_date')['profit'].agg(['mean', 'std', 'min', 'max']).fillna(0)
        daily_mean_profit = daily_profits['mean'].mean()
        daily_std_profit = daily_profits['mean'].std() if len(daily_profits) > 1 else 0
        daily_min_profit = daily_profits['min'].min()
        daily_max_profit = daily_profits['max'].max()
        
        # Weekly statistics
        weekly_profits = trades_df.groupby('week')['profit'].agg(['mean', 'std', 'min', 'max']).fillna(0)
        weekly_mean_profit = weekly_profits['mean'].mean()
        weekly_std_profit = weekly_profits['mean'].std() if len(weekly_profits) > 1 else 0
        weekly_min_profit = weekly_profits['min'].min()
        weekly_max_profit = weekly_profits['max'].max()
    else:
        daily_mean_profit = daily_std_profit = daily_min_profit = daily_max_profit = 0
        weekly_mean_profit = weekly_std_profit = weekly_min_profit = weekly_max_profit = 0
    
    results = {
        'total_profit': total_profit,
        'number_of_trades': num_trades,
        'win_rate': win_rate,
        'trades': trades,
        'final_open_position': {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'shares': position_size
        } if in_position else None,
        'daily_mean_profit': daily_mean_profit,
        'daily_std_profit': daily_std_profit,
        'daily_min_profit': daily_min_profit,
        'daily_max_profit': daily_max_profit,
        'weekly_mean_profit': weekly_mean_profit,
        'weekly_std_profit': weekly_std_profit,
        'weekly_min_profit': weekly_min_profit,
        'weekly_max_profit': weekly_max_profit
    }
    
    return results

def backtest_many(start_date="2025-02-01",end_date="2025-02-24"):

    mdf = pl.read_csv("ins_data/cop_nasd_5m.csv")
    mdf = mdf.with_columns((pl.col("Datetime").str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")).alias("Datetime"))
    mdf = mdf.filter((pl.col("Datetime")>=str_to_dt(start_date)) & (pl.col("Datetime")<=str_to_dt(end_date)))
    mdf = get_high_quality_data(mdf)
    mdf = pl.concat(fill_missing_datetime(mdf.filter(pl.col("ticker")=="NOVO-B.CO"),[mdf.filter(pl.col("ticker")==tick) for tick in list(mdf["ticker"].unique())]))
    backtest_results = []
    take_profit_pct=0.0075
    stop_loss_pct=0.025
    low_change_pct=-0.00086
    for tick in list(mdf["ticker"].unique()):
        df = mdf.filter(pl.col("ticker")==tick)
        df = df.to_pandas()
        df = df.set_index('Datetime')
        results = backtest_trading_strategy(df, take_profit_pct=take_profit_pct, stop_loss_pct=stop_loss_pct, low_change_pct=low_change_pct, investment_amount=10000)
        del results["trades"]
        del results["final_open_position"]
        results["ticker"]=tick
        results["take_profit_pct"]=take_profit_pct
        results["stop_loss_pct"]=stop_loss_pct
        results["low_change_pct"]=low_change_pct
        print (tick)
        print(f"Total Profit: DKK{results['total_profit']:.2f}")
        print(f"Number of Trades: {results['number_of_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print ("\n")
        backtest_results.append(results)
    #print (backtest_results)
    backtest_results = pl.DataFrame(backtest_results)
    #print (backtest_results)
    backtest_results.write_csv("backtesting_data/cop_nasd_5m_backtest.csv")

    print (backtest_results["total_profit"].mean())

# Example usage:
# Assuming your dataframe 'df' has a datetime index and 'high'/'low' columns

prepare_train_data()
sys.exit()

backtest_many()
sys.exit()

# Sample dataframe structure:
df = pl.read_csv("ins_data/cop_nasd_5m.csv")
df = df.with_columns((pl.col("Datetime").str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")).alias("Datetime"))
df = df.filter(pl.col("ticker")=="NOVO-B.CO")
df = df.to_pandas()
df = df.set_index('Datetime')

# Run the backtest
results = backtest_trading_strategy(df, take_profit_pct=0.015, stop_loss_pct=0.035, low_change_pct=0.001, investment_amount=10000)
print(f"Total Profit: DKK{results['total_profit']:.2f}")
print(f"Number of Trades: {results['number_of_trades']}")
print(f"Win Rate: {results['win_rate']:.2%}")
