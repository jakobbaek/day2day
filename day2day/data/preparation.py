"""Training data preparation module."""

import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from ..config.settings import settings
from .datetime_utils import DateTimeStandardizer

logger = logging.getLogger(__name__)

class DataPreparator:
    """Prepares raw market data for model training."""
    
    def __init__(self):
        self.settings = settings
        self.prediction_horizon = settings.prediction_horizon_minutes
        self.datetime_standardizer = DateTimeStandardizer()
    
    def load_raw_data(self, filename: str, standardize_datetime: bool = True) -> pl.DataFrame:
        """
        Load raw market data from CSV file with optional datetime standardization.
        
        Args:
            filename: Name of the CSV file in raw data directory
            standardize_datetime: Whether to standardize datetime to GMT and create complete timeline
            
        Returns:
            Polars DataFrame with raw data
        """
        file_path = settings.get_raw_data_file(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")
        
        df = pl.read_csv(file_path)
        
        # Ensure datetime column is properly formatted
        df = df.with_columns(
            pl.col("datetime").str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")
        )
        
        if standardize_datetime:
            logger.info("Standardizing datetime and creating complete timeline")
            df = self.datetime_standardizer.standardize_all_instruments(df)
        
        return df
    
    def filter_high_quality_data(self, df: pl.DataFrame, min_daily_observations: int = 25) -> pl.DataFrame:
        """
        Filter data to keep only high-quality instruments with sufficient observations.
        
        Args:
            df: Input DataFrame
            min_daily_observations: Minimum observations per day to keep instrument
            
        Returns:
            Filtered DataFrame
        """
        # Extract date from datetime
        df = df.with_columns(
            pl.col("datetime").dt.date().alias("date")
        )
        
        # Group by ticker and date, count observations
        daily_counts = (df
            .group_by(["ticker", "date"])
            .agg(pl.count().alias("count"))
        )
        
        # Calculate mean daily observations per ticker
        mean_counts = (daily_counts
            .group_by("ticker")
            .agg(pl.col("count").mean().alias("mean_count"))
        )
        
        # Filter tickers with sufficient data quality
        quality_tickers = (mean_counts
            .filter(pl.col("mean_count") >= min_daily_observations)
            .select("ticker")
            .to_series()
            .to_list()
        )
        
        logger.info(f"Kept {len(quality_tickers)} high-quality instruments out of {len(mean_counts)}")
        
        return df.filter(pl.col("ticker").is_in(quality_tickers)).drop("date")
    
    def fill_missing_timestamps(self, df: pl.DataFrame, reference_ticker: str) -> pl.DataFrame:
        """
        Fill missing timestamps using forward fill strategy.
        
        NOTE: This method is deprecated. Use load_raw_data() with standardize_datetime=True 
        for proper datetime standardization and gap filling.
        
        Args:
            df: Input DataFrame
            reference_ticker: Ticker to use as reference for timestamps
            
        Returns:
            DataFrame with filled timestamps
        """
        logger.warning("fill_missing_timestamps is deprecated. Use datetime standardization instead.")
        
        # If data is already standardized, return as-is
        if self._is_data_standardized(df):
            logger.info("Data appears to be already standardized, returning as-is")
            return df
        
        # Get reference timestamps
        reference_df = df.filter(pl.col("ticker") == reference_ticker)
        
        if reference_df.is_empty():
            logger.warning(f"Reference ticker {reference_ticker} not found, using first available ticker")
            reference_df = df.filter(pl.col("ticker") == df.select("ticker").unique().to_series()[0])
        
        # Get all unique timestamps
        all_timestamps = (df
            .select("datetime")
            .unique()
            .sort("datetime")
        )
        
        # Get unique tickers
        tickers = df.select("ticker").unique().to_series().to_list()
        
        filled_dfs = []
        
        for ticker in tickers:
            ticker_df = df.filter(pl.col("ticker") == ticker)
            
            # Join with all timestamps and forward fill
            filled_df = (all_timestamps
                .join(ticker_df, on="datetime", how="left")
                .fill_null(strategy="forward")
                .filter(pl.col("ticker").is_not_null())
            )
            
            filled_dfs.append(filled_df)
        
        return pl.concat(filled_dfs)
    
    def _is_data_standardized(self, df: pl.DataFrame) -> bool:
        """Check if data appears to be already standardized."""
        # Simple heuristic: check if we have consistent 1-minute intervals
        if df.is_empty():
            return False
        
        # Check one ticker for consistent intervals
        ticker = df.select("ticker").unique().to_series()[0]
        ticker_data = df.filter(pl.col("ticker") == ticker).sort("datetime")
        
        if len(ticker_data) < 2:
            return False
        
        # Check if intervals are mostly 1 minute
        intervals = ticker_data.select(
            (pl.col("datetime").diff().dt.total_seconds() / 60).alias("interval_minutes")
        ).drop_nulls()
        
        if intervals.is_empty():
            return False
        
        # If most intervals are 1 minute, consider it standardized
        one_minute_ratio = (intervals.filter(pl.col("interval_minutes") == 1.0).height / 
                           intervals.height)
        
        return one_minute_ratio > 0.8
    
    def create_wide_format(self, df: pl.DataFrame, value_columns: List[str] = None) -> pl.DataFrame:
        """
        Convert long format data to wide format suitable for model training.
        
        Args:
            df: Input DataFrame in long format
            value_columns: Columns to pivot (default: ["high", "low", "open", "close"])
            
        Returns:
            DataFrame in wide format
        """
        if value_columns is None:
            value_columns = ["high", "low", "open", "close"]
        
        # Get unique tickers
        tickers = df.select("ticker").unique().to_series().to_list()
        
        # Start with datetime column
        reference_df = df.filter(pl.col("ticker") == tickers[0]).select("datetime")
        
        # Join data for each ticker
        for ticker in tickers:
            ticker_df = df.filter(pl.col("ticker") == ticker)
            
            # Rename columns to include ticker
            rename_dict = {col: f"{col}_{ticker}" for col in value_columns}
            ticker_df = ticker_df.select(["datetime"] + value_columns).rename(rename_dict)
            
            # Join with reference
            reference_df = reference_df.join(ticker_df, on="datetime", how="left")
        
        return reference_df
    
    def add_percentage_features(self, df: pl.DataFrame, 
                              target_columns: List[str] = None,
                              method: str = "previous_day_close") -> pl.DataFrame:
        """
        Add percentage change features to the dataset.
        
        Args:
            df: Input DataFrame
            target_columns: Columns to calculate percentage changes for
            method: Method for calculating percentage changes ("previous_day_close", "previous_value")
            
        Returns:
            DataFrame with percentage features
        """
        if target_columns is None:
            target_columns = [col for col in df.columns if col != "datetime"]
        
        result_df = df.clone()
        
        if method == "previous_day_close":
            # Calculate percentage relative to previous day's close
            result_df = result_df.with_columns(
                pl.col("datetime").dt.date().alias("date")
            )
            
            for col in target_columns:
                if "close_" in col:
                    # Use this as the reference close price
                    ticker = col.split("_", 1)[1]
                    close_col = f"close_{ticker}"
                    
                    if close_col in df.columns:
                        # Get previous day's close
                        daily_close = (result_df
                            .group_by("date")
                            .agg(pl.col(close_col).last().alias(f"prev_close_{ticker}"))
                            .with_columns(pl.col("date") + pl.duration(days=1))
                        )
                        
                        # Join and calculate percentage
                        result_df = result_df.join(daily_close, on="date", how="left")
                        
                        # Calculate percentage change for all columns related to this ticker
                        for value_col in target_columns:
                            if value_col.endswith(f"_{ticker}"):
                                pct_col = f"{value_col}_pct"
                                result_df = result_df.with_columns(
                                    ((pl.col(value_col) - pl.col(f"prev_close_{ticker}")) / 
                                     pl.col(f"prev_close_{ticker}")).alias(pct_col)
                                )
                        
                        # Clean up temporary column
                        result_df = result_df.drop(f"prev_close_{ticker}")
            
            result_df = result_df.drop("date")
        
        elif method == "previous_value":
            # Calculate percentage relative to previous value
            for col in target_columns:
                pct_col = f"{col}_pct"
                result_df = result_df.with_columns(
                    (pl.col(col).pct_change()).alias(pct_col)
                )
        
        return result_df
    
    def add_historical_features(self, df: pl.DataFrame, 
                              target_columns: List[str] = None,
                              periods: Dict[str, int] = None) -> pl.DataFrame:
        """
        Add historical mean features.
        
        Args:
            df: Input DataFrame
            target_columns: Columns to calculate historical means for
            periods: Dictionary of period names to number of days
            
        Returns:
            DataFrame with historical features
        """
        if target_columns is None:
            target_columns = [col for col in df.columns if col != "datetime"]
        
        if periods is None:
            periods = {
                "2y": 730,
                "1y": 365,
                "3m": 90,
                "10d": 10,
                "5d": 5
            }
        
        result_df = df.clone()
        
        for period_name, days in periods.items():
            window_size = days * 288  # Approximate 5-minute intervals per day
            
            for col in target_columns:
                if col != "datetime":
                    feature_name = f"{col}_mean_{period_name}"
                    result_df = result_df.with_columns(
                        pl.col(col).rolling_mean(window_size).alias(feature_name)
                    )
        
        return result_df
    
    def create_lagged_features(self, df: pl.DataFrame, 
                             target_columns: List[str] = None,
                             max_lag: int = 12) -> pl.DataFrame:
        """
        Create lagged features for time series prediction.
        
        Args:
            df: Input DataFrame
            target_columns: Columns to create lags for
            max_lag: Maximum lag to create
            
        Returns:
            DataFrame with lagged features
        """
        if target_columns is None:
            target_columns = [col for col in df.columns if col != "datetime"]
        
        result_df = df.clone()
        lags = [1, 5, 12] if max_lag >= 12 else [1, min(5, max_lag)]
        
        for lag in lags:
            if lag <= max_lag:
                for col in target_columns:
                    if col != "datetime":
                        lag_col = f"{col}_lag{lag}"
                        result_df = result_df.with_columns(
                            pl.col(col).shift(lag).alias(lag_col)
                        )
        
        return result_df
    
    def create_target_variable(self, df: pl.DataFrame, 
                             target_column: str,
                             horizon: int = None) -> pl.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            df: Input DataFrame
            target_column: Column name for target variable
            horizon: Prediction horizon in time steps
            
        Returns:
            DataFrame with target variable
        """
        if horizon is None:
            horizon = self.prediction_horizon
        
        result_df = df.clone()
        
        # Create future target
        result_df = result_df.with_columns(
            pl.col(target_column).shift(-horizon).alias("target")
        )
        
        return result_df
    
    def add_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        return df.with_columns([
            pl.col("datetime").dt.hour().alias("hour"),
            pl.col("datetime").dt.minute().alias("minute"),
            pl.col("datetime").dt.weekday().alias("dayofweek"),
            pl.col("datetime").dt.day().alias("day"),
            pl.col("datetime").dt.month().alias("month")
        ])
    
    def prepare_training_data(self, 
                            raw_data_file: str,
                            output_title: str,
                            target_instrument: str,
                            target_price_type: str = "high",
                            use_percentage_features: bool = True,
                            use_raw_prices: bool = True,
                            selected_instruments: List[str] = None,
                            add_historical_means: bool = True,
                            max_lag: int = 12,
                            min_daily_observations: int = 25) -> str:
        """
        Prepare training data according to specifications.
        
        Args:
            raw_data_file: Name of raw data file
            output_title: Title for output file
            target_instrument: Target instrument for prediction
            target_price_type: Type of price to predict (NOTE: Always forced to "high" as per specifications)
            use_percentage_features: Whether to add percentage features
            use_raw_prices: Whether to include raw price features
            selected_instruments: List of instruments to include (None for all)
            add_historical_means: Whether to add historical mean features
            max_lag: Maximum lag for lagged features
            min_daily_observations: Minimum daily observations for quality filter
            
        Returns:
            Path to output file
        """
        logger.info(f"Preparing training data: {output_title}")
        
        # Load raw data with datetime standardization (GMT conversion and complete timeline)
        df = self.load_raw_data(raw_data_file, standardize_datetime=True)
        
        # Filter high-quality data
        df = self.filter_high_quality_data(df, min_daily_observations)
        
        # Filter selected instruments
        if selected_instruments:
            df = df.filter(pl.col("ticker").is_in(selected_instruments))
        
        # Note: Datetime standardization already handles missing timestamps and forward filling
        
        # Convert to wide format
        value_columns = ["high", "low", "open", "close"]
        df = self.create_wide_format(df, value_columns)
        
        # Add percentage features if requested
        if use_percentage_features:
            df = self.add_percentage_features(df)
        
        # Add historical features if requested
        if add_historical_means:
            df = self.add_historical_features(df)
        
        # Create lagged features
        df = self.create_lagged_features(df, max_lag=max_lag)
        
        # Add time features
        df = self.add_time_features(df)
        
        # Create target variable - always use high price as per specifications
        target_price_type = "high"  # Force high price as target
        target_column = f"{target_price_type}_{target_instrument}"
        if target_column not in df.columns:
            raise ValueError(f"Target column {target_column} not found in data")
        
        logger.info(f"Creating target variable using HIGH price: {target_column}")
        df = self.create_target_variable(df, target_column)
        
        # Remove rows with missing target
        df = df.filter(pl.col("target").is_not_null())
        
        # Save prepared data
        output_file = f"{output_title}.csv"
        output_path = settings.get_processed_data_file(output_file)
        
        df.write_csv(output_path)
        logger.info(f"Saved training data to {output_path} ({len(df)} rows)")
        
        return str(output_path)
    
    def update_training_data(self, 
                           title: str,
                           new_raw_data_file: str) -> str:
        """
        Update existing training data with new raw data.
        
        Args:
            title: Title of existing training data
            new_raw_data_file: New raw data file to incorporate
            
        Returns:
            Path to updated training data file
        """
        # Load existing configuration
        config_file = f"{title}_config.json"
        config = settings.load_config(config_file)
        
        if not config:
            raise ValueError(f"Configuration for {title} not found")
        
        # Re-prepare data with new raw data
        return self.prepare_training_data(
            raw_data_file=new_raw_data_file,
            output_title=title,
            **config
        )