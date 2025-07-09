import pandas as pd
import numpy as np
import polars as pl
import sys
from datetime import datetime

def str_to_dt(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d")

def prepare_train_data():

    #mdf = pl.concat([pl.read_csv("ins_data/cop_nasd_5m.csv"),pl.read_csv("ins_data/forex.csv")])
    mdf = pl.concat([pl.read_csv("ins_data/cop_saxo_1m.csv")])
    mdf = mdf.with_columns((pl.col("datetime").str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")).alias("datetime"))
    mdf = get_high_quality_data(mdf)
    tickers = list(mdf["ticker"].unique())
    mdf = pl.concat(fill_missing_datetime(mdf.filter(pl.col("ticker")=="NOVO-B.CO"),[mdf.filter(pl.col("ticker")==tick) for tick in tickers]))

    new_df = mdf.filter(pl.col("ticker")==tickers[0]).select(["High","Low","Open","Close","datetime"]).rename({"High":"High_{}".format(tickers[0]),"Low":"Low_{}".format(tickers[0]),"Open":"Open_{}".format(tickers[0]),"Close":"Close_{}".format(tickers[0])}) 
    for tick in tickers:
        if  tick != tickers[0]:
            ndf = mdf.filter(pl.col("ticker")==tick).select(["High","Low","Open","Close","datetime"]).rename({"High":"High_{}".format(tick),"Low":"Low_{}".format(tick),"Open":"Open_{}".format(tick),"Close":"Close_{}".format(tick)})
            new_df = new_df.join(ndf,on="datetime",how="left")
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
                   .with_columns(pl.col('datetime').cast(pl.Datetime))
                   .sort('datetime'))
    
    # Collect all unique datetime values from all dataframes
    all_dfs = [reference_df] + list(dataframes)
    all_dates = (pl.concat([df.select('datetime') for df in all_dfs])
                .unique()
                .sort('datetime'))
    
    # Process each input dataframe
    result_dfs = []
    for df in dataframes:
        # Convert to Polars if needed, ensure datetime format and sort
        if not isinstance(df, pl.DataFrame):
            df = pl.DataFrame(df)
        df = (df
             .with_columns(pl.col('datetime').cast(pl.Datetime))
             .sort('datetime'))
        
        # Join with all unique dates and forward fill
        filled_df = (all_dates
                    .join(df, on='datetime', how='left')
                    .fill_null(strategy='forward'))
        
        result_dfs.append(filled_df)
    
    return result_dfs

def get_high_quality_data(df):

    # Extract date from datetime
    df = df.with_columns(
        pl.col("datetime").dt.date().alias("Date")
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
    df = df.filter(pl.col("ticker").is_in(mean_counts.filter(pl.col("count")>25.0)["ticker"].to_list()))
    return df

prepare_train_data()
sys.exit()

