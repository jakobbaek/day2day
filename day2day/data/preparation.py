"""Training data preparation module."""

import polars as pl
import pandas as pd
import numpy as np
import time
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
    
    def filter_by_trading_volume(self, df: pl.DataFrame, volume_fraction: float = 1.0) -> pl.DataFrame:
        """
        Filter instruments based on trading volume (inferred from price estimates per day).
        
        Args:
            df: Input DataFrame
            volume_fraction: Fraction of most traded stocks to include (0.0 to 1.0)
                           e.g., 0.25 = top 25% most traded stocks
                           
        Returns:
            Filtered DataFrame with only the most traded instruments
        """
        if volume_fraction <= 0.0 or volume_fraction > 1.0:
            raise ValueError("volume_fraction must be between 0.0 and 1.0")
        
        logger.info(f"Filtering instruments by trading volume (top {volume_fraction*100:.1f}%)")
        
        # Extract date from datetime
        df_with_date = df.with_columns(
            pl.col("datetime").dt.date().alias("date")
        )
        
        # Calculate daily observations per ticker
        daily_counts = (df_with_date
            .group_by(["ticker", "date"])
            .agg(pl.count().alias("daily_count"))
        )
        
        # Calculate mean daily observations per ticker (proxy for trading volume)
        mean_trading_volume = (daily_counts
            .group_by("ticker")
            .agg(pl.col("daily_count").mean().alias("avg_daily_observations"))
            .sort("avg_daily_observations", descending=True)
        )
        
        # Calculate how many instruments to keep
        total_instruments = len(mean_trading_volume)
        keep_count = max(1, int(total_instruments * volume_fraction))
        
        # Select top instruments by trading volume
        top_instruments = (mean_trading_volume
            .head(keep_count)
            .select("ticker")
            .to_series()
            .to_list()
        )
        
        logger.info(f"Selected {keep_count} most traded instruments out of {total_instruments}")
        logger.info(f"Top instruments: {top_instruments[:5]}{'...' if len(top_instruments) > 5 else ''}")
        
        # Log trading volume statistics
        selected_stats = mean_trading_volume.head(keep_count)
        max_volume = selected_stats.select("avg_daily_observations").max().item()
        min_volume = selected_stats.select("avg_daily_observations").min().item()
        logger.info(f"Trading volume range: {min_volume:.1f} - {max_volume:.1f} avg daily observations")
        
        return df.filter(pl.col("ticker").is_in(top_instruments))
    
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
                        # Get daily close prices for this ticker, sorted by date
                        daily_close = (result_df
                            .group_by("date")
                            .agg(pl.col(close_col).last().alias(f"daily_close_{ticker}"))
                            .sort("date")
                        )
                        
                        # Create previous trading day mapping
                        # Shift the close prices by 1 position to get previous trading day's close
                        daily_close_with_prev = daily_close.with_columns([
                            pl.col(f"daily_close_{ticker}").shift(1).alias(f"prev_close_{ticker}")
                        ]).select(["date", f"prev_close_{ticker}"])
                        
                        # Join and calculate percentage
                        result_df = result_df.join(daily_close_with_prev, on="date", how="left")
                        
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
            target_columns: Columns to calculate historical means for (defaults to Open prices only)
            periods: Dictionary of period names to number of days (defaults to 3m and 10d only)
            
        Returns:
            DataFrame with historical features
        """
        if target_columns is None:
            # Default to only Open prices for efficiency
            target_columns = [col for col in df.columns if col.startswith("open_") and col != "datetime"]
            logger.info(f"Using default target columns (Open prices only): {len(target_columns)} columns")
        
        if periods is None:
            periods = {
                "3m": 90,
                "10d": 10
            }
            logger.info(f"Using default periods (optimized): {list(periods.keys())}")
        
        result_df = df.clone()
        
        logger.info(f"Adding historical features for {len(target_columns)} columns across {len(periods)} periods")
        logger.info("Optimizing by calculating daily means first, then joining back to minute-level data")
        
        # Add date column for grouping
        result_df = result_df.with_columns(
            pl.col("datetime").dt.date().alias("date")
        )
        
        for period_name, days in periods.items():
            logger.info(f"Processing {period_name} period ({days} days lookback)")
            period_start = time.time()
            
            # Step 1: Create daily aggregates for ALL instruments at once
            logger.info(f"Calculating daily means for all target columns")
            
            # Prepare aggregation expressions for all target columns
            agg_exprs = []
            for col in target_columns:
                if col != "datetime":
                    agg_exprs.append(pl.col(col).mean().alias(f"daily_{col}"))
            
            # Calculate daily means for all instruments in one operation
            daily_aggregates = (result_df
                .group_by("date")
                .agg(agg_exprs)
                .sort("date")
            )
            
            # Step 2: Calculate rolling means for all columns
            rolling_exprs = []
            for col in target_columns:
                if col != "datetime":
                    daily_col = f"daily_{col}"
                    feature_name = f"{col}_mean_{period_name}"
                    rolling_exprs.append(
                        pl.col(daily_col).rolling_mean(days).alias(feature_name)
                    )
            
            daily_features = daily_aggregates.with_columns(rolling_exprs)
            
            # Select only the date and the rolling mean features
            feature_cols = ["date"] + [f"{col}_mean_{period_name}" for col in target_columns if col != "datetime"]
            daily_features = daily_features.select(feature_cols)
            
            # Step 3: Join back to the main dataframe
            result_df = result_df.join(daily_features, on="date", how="left")
            
            features_added = len([col for col in target_columns if col != "datetime"])
            logger.info(f"✓ {period_name} period completed in {time.time() - period_start:.2f}s - Added {features_added} features")
        
        # Remove temporary date column
        result_df = result_df.drop("date")
        
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
                            use_percentage_features: bool = False,
                            selected_instruments: List[str] = None,
                            add_historical_means: bool = True,
                            max_lag: int = 12,
                            min_daily_observations: int = 25,
                            volume_fraction: float = 1.0) -> str:
        """
        Prepare training data according to specifications.
        
        Args:
            raw_data_file: Name of raw data file
            output_title: Title for output file
            target_instrument: Target instrument for prediction
            target_price_type: Type of price to predict (NOTE: Always forced to "high" as per specifications)
            use_percentage_features: If True, use percentage changes instead of raw prices
                                   If False (default), use raw price values
                                   Note: Raw and percentage features are mutually exclusive
            selected_instruments: List of instruments to include (None for all)
            add_historical_means: Whether to add historical mean features
            max_lag: Maximum lag for lagged features
            min_daily_observations: Minimum daily observations for quality filter
            volume_fraction: Fraction of most traded stocks to include (0.0 to 1.0)
                           e.g., 0.25 = top 25% most traded stocks
            
        Returns:
            Path to output file
        """
        logger.info(f"=== STARTING DATA PREPARATION: {output_title} ===")
        start_time = time.time()
        
        # Step 1: Load raw data with datetime standardization
        logger.info("Step 1/9: Loading raw data with datetime standardization...")
        step_start = time.time()
        df = self.load_raw_data(raw_data_file, standardize_datetime=True)
        logger.info(f"✓ Step 1 completed in {time.time() - step_start:.2f}s - Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Step 2: Filter high-quality data (currently disabled)
        logger.info("Step 2/9: Filtering high-quality data (currently disabled)")
        #df = self.filter_high_quality_data(df, min_daily_observations)
        
        # Step 3: Filter by trading volume
        if volume_fraction < 1.0:
            logger.info(f"Step 3/9: Filtering by trading volume (top {volume_fraction*100:.1f}%)...")
            step_start = time.time()
            df = self.filter_by_trading_volume(df, volume_fraction)
            logger.info(f"✓ Step 3 completed in {time.time() - step_start:.2f}s - Reduced to {len(df)} rows")
        else:
            logger.info("Step 3/9: Skipping trading volume filter (volume_fraction=1.0)")
        
        # Step 4: Filter selected instruments
        if selected_instruments:
            logger.info(f"Step 4/9: Filtering selected instruments ({len(selected_instruments)} instruments)...")
            step_start = time.time()
            df = df.filter(pl.col("ticker").is_in(selected_instruments))
            logger.info(f"✓ Step 4 completed in {time.time() - step_start:.2f}s - Reduced to {len(df)} rows")
        else:
            logger.info("Step 4/9: Skipping instrument selection (using all instruments)")
        
        # Step 5: Convert to wide format
        logger.info("Step 5/9: Converting to wide format...")
        step_start = time.time()
        value_columns = ["high", "low", "open", "close"]
        df = self.create_wide_format(df, value_columns)
        logger.info(f"✓ Step 5 completed in {time.time() - step_start:.2f}s - Wide format: {len(df)} rows, {len(df.columns)} columns")
        
        # Step 6: Handle price feature type selection (raw vs percentage)
        if use_percentage_features:
            logger.info("Step 6/9: Converting raw prices to percentage features...")
            step_start = time.time()
            
            # Calculate percentage features (this adds _pct columns)
            df = self.add_percentage_features(df)
            
            # Replace raw price columns with percentage columns (same column names)
            price_columns = [col for col in df.columns if any(col.startswith(f"{price}_") for price in value_columns)]
            
            for col in price_columns:
                pct_col = f"{col}_pct"
                if pct_col in df.columns:
                    # Replace raw price column with percentage values, keeping same column name
                    df = df.drop(col).rename({pct_col: col})
            
            logger.info(f"✓ Step 6 completed in {time.time() - step_start:.2f}s - Converted to percentage features: {len(df.columns)} columns")
            logger.info("Price columns now contain percentage changes (same column names)")
        else:
            logger.info("Step 6/9: Using raw prices (default)")
            logger.info("Price columns contain raw price values")
        
        # Step 7: Add historical features
        if add_historical_means:
            logger.info("Step 7/9: Adding historical mean features...")
            step_start = time.time()
            df = self.add_historical_features(df)
            logger.info(f"✓ Step 7 completed in {time.time() - step_start:.2f}s - Added historical features: {len(df.columns)} columns")
        else:
            logger.info("Step 7/9: Skipping historical features")
        
        # Step 8: Create lagged features
        logger.info(f"Step 8/9: Creating lagged features (max_lag={max_lag})...")
        step_start = time.time()
        df = self.create_lagged_features(df, max_lag=max_lag)
        logger.info(f"✓ Step 8 completed in {time.time() - step_start:.2f}s - Added lagged features: {len(df.columns)} columns")
        
        # Add time features
        logger.info("Step 8.5/9: Adding time-based features...")
        step_start = time.time()
        df = self.add_time_features(df)
        logger.info(f"✓ Step 8.5 completed in {time.time() - step_start:.2f}s - Added time features: {len(df.columns)} columns")
        
        # Step 9: Create target variable and finalize
        logger.info("Step 9/9: Creating target variable and finalizing...")
        step_start = time.time()
        
        # Create target variable - always use high price as per specifications
        target_price_type = "high"  # Force high price as target
        target_column = f"{target_price_type}_{target_instrument}"
        
        # Debug: Show available columns to help with troubleshooting
        available_high_cols = [col for col in df.columns if col.startswith("high_")]
        logger.info(f"Available HIGH columns: {available_high_cols[:10]}{'...' if len(available_high_cols) > 10 else ''}")
        logger.info(f"Looking for target column: {target_column}")
        
        if target_column not in df.columns:
            # Try to find similar column names
            similar_cols = [col for col in df.columns if target_instrument in col and "high" in col]
            if similar_cols:
                logger.error(f"Target column '{target_column}' not found, but found similar: {similar_cols}")
                logger.error("Possible ticker format mismatch. Check if ticker in data matches target_instrument parameter.")
            else:
                logger.error(f"No columns found containing '{target_instrument}' and 'high'")
                logger.error(f"Available ticker formats in data: {[col.split('_')[-1] for col in available_high_cols[:5]]}")
            raise ValueError(f"Target column {target_column} not found in data")
        
        logger.info(f"Creating target variable using HIGH price: {target_column}")
        df = self.create_target_variable(df, target_column)
        
        # Remove rows with missing target
        initial_rows = len(df)
        df = df.filter(pl.col("target").is_not_null())
        final_rows = len(df)
        logger.info(f"Removed {initial_rows - final_rows} rows with missing target values")
        
        # Save prepared data
        output_file = f"{output_title}.csv"
        output_path = settings.get_processed_data_file(output_file)
        
        df.write_csv(output_path)
        logger.info(f"✓ Step 9 completed in {time.time() - step_start:.2f}s - Saved to {output_path}")
        
        total_time = time.time() - start_time
        logger.info(f"=== DATA PREPARATION COMPLETED in {total_time:.2f}s ===")
        logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
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