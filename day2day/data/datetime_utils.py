"""Datetime utilities for market data standardization."""

import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class DateTimeStandardizer:
    """Handles datetime standardization and market hours inference."""
    
    def __init__(self):
        self.market_hours_cache: Dict[str, Tuple[time, time]] = {}
        
    def standardize_to_gmt(self, df: pl.DataFrame, datetime_col: str = "datetime") -> pl.DataFrame:
        """
        Convert datetime column to GMT timezone.
        
        Args:
            df: Input DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with GMT standardized datetime
        """
        logger.info("Standardizing datetimes to GMT")
        
        # Check if datetime column is already parsed or needs parsing
        datetime_dtype = df.select(pl.col(datetime_col)).dtypes[0]
        
        if datetime_dtype == pl.String or datetime_dtype == pl.Utf8:
            # Parse string datetime
            df = df.with_columns(
                pl.col(datetime_col).str.slice(0, length=19).str.to_datetime("%Y-%m-%dT%H:%M:%S")
            )
        elif datetime_dtype != pl.Datetime:
            # Convert other types to datetime if needed
            df = df.with_columns(
                pl.col(datetime_col).cast(pl.Datetime)
            )
        
        # Convert to UTC/GMT if timezone info is available
        # For Danish market data, assume CET/CEST timezone
        # CET is UTC+1, CEST is UTC+2 (daylight saving)
        df = df.with_columns(
            pl.col(datetime_col).map_elements(
                lambda dt: self._convert_danish_to_gmt(dt),
                return_dtype=pl.Datetime
            ).alias(f"{datetime_col}_gmt")
        )
        
        # Replace original datetime column
        df = df.drop(datetime_col).rename({f"{datetime_col}_gmt": datetime_col})
        
        return df
    
    def _convert_danish_to_gmt(self, dt: datetime) -> datetime:
        """
        Convert Danish timezone (CET/CEST) to GMT.
        
        Args:
            dt: Datetime in Danish timezone
            
        Returns:
            Datetime in GMT
        """
        if dt is None:
            return None
            
        # Simple conversion assuming CET (UTC+1) for winter, CEST (UTC+2) for summer
        # DST in Europe: last Sunday in March to last Sunday in October
        year = dt.year
        
        # Calculate DST boundaries
        march_last_sunday = self._last_sunday_of_month(year, 3)
        october_last_sunday = self._last_sunday_of_month(year, 10)
        
        # Check if in daylight saving time
        if march_last_sunday <= dt.replace(tzinfo=None) < october_last_sunday:
            # CEST (UTC+2)
            gmt_dt = dt - timedelta(hours=2)
        else:
            # CET (UTC+1)
            gmt_dt = dt - timedelta(hours=1)
        
        return gmt_dt
    
    def _last_sunday_of_month(self, year: int, month: int) -> datetime:
        """Find the last Sunday of a given month."""
        # Start with the last day of the month
        if month == 12:
            last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            last_day = datetime(year, month + 1, 1) - timedelta(days=1)
        
        # Find the last Sunday
        days_since_sunday = last_day.weekday() + 1
        if days_since_sunday == 7:
            days_since_sunday = 0
        
        last_sunday = last_day - timedelta(days=days_since_sunday)
        return last_sunday
    
    def infer_market_hours(self, df: pl.DataFrame, ticker: str = None) -> Tuple[time, time]:
        """
        Infer market opening hours from ALL instruments in the data.
        All instruments share the same market hours - the earliest opening time 
        and latest closing time across all instruments for each day.
        
        Args:
            df: DataFrame with market data for all instruments
            ticker: Ticker symbol (used for caching, but hours are calculated globally)
            
        Returns:
            Tuple of (market_open_time, market_close_time)
        """
        cache_key = "global_market_hours"
        if cache_key in self.market_hours_cache:
            return self.market_hours_cache[cache_key]
        
        logger.info("Inferring global market hours from all instruments")
        
        if df.is_empty():
            # Default to Copenhagen Stock Exchange hours in GMT
            default_open = time(8, 0)  # Conservative estimate
            default_close = time(16, 0)
            self.market_hours_cache[cache_key] = (default_open, default_close)
            return default_open, default_close
        
        # Extract time and date components from datetime for all data
        df_with_time = df.with_columns([
            pl.col("datetime").dt.time().alias("time_only"),
            pl.col("datetime").dt.date().alias("date_only")
        ])
        
        # Group by date and find absolute min/max times across ALL instruments for each day
        daily_global_hours = (df_with_time
            .group_by("date_only")
            .agg([
                pl.col("time_only").min().alias("global_day_open"),
                pl.col("time_only").max().alias("global_day_close"),
                pl.col("time_only").count().alias("total_observations")
            ])
            .filter(pl.col("total_observations") > 50)  # Filter out days with too few observations
        )
        
        if daily_global_hours.is_empty():
            # Fallback to default hours
            default_open = time(8, 0)
            default_close = time(16, 0)
            self.market_hours_cache[cache_key] = (default_open, default_close)
            return default_open, default_close
        
        # Calculate most common global opening and closing times
        open_times = daily_global_hours.select("global_day_open").to_series().to_list()
        close_times = daily_global_hours.select("global_day_close").to_series().to_list()
        
        # DEBUG: Show what times we're seeing in the data
        logger.debug(f"Sample opening times from data: {open_times[:10]}")
        logger.debug(f"Sample closing times from data: {close_times[:10]}")
        
        # Use mode (most common time) or median if mode is not clear
        market_open = self._calculate_typical_time(open_times)
        market_close = self._calculate_typical_time(close_times)
        
        # CRITICAL FIX: Danish market hours should be ~7:00-15:00 GMT
        # If we're getting 5:00-13:00, there's a timezone issue
        if market_open.hour < 6:
            logger.error(f"CRITICAL: Inferred market open ({market_open}) too early!")
            logger.error("Danish market opens around 9:00 CET = 7:00-8:00 GMT")
            logger.error("This suggests timezone conversion error in raw data or inference")
            
            # Apply a correction - shift to more realistic hours
            from datetime import time
            corrected_open = time(7, 0)  # 7:00 GMT = 8:00/9:00 CET
            corrected_close = time(15, 0)  # 15:00 GMT = 16:00/17:00 CET
            
            logger.warning(f"APPLYING CORRECTION: Using {corrected_open} - {corrected_close} GMT")
            logger.warning("This should prevent data bleeding from incorrect early timeline slots")
            
            market_open = corrected_open
            market_close = corrected_close
        
        self.market_hours_cache[cache_key] = (market_open, market_close)
        
        logger.info(f"Inferred global market hours: {market_open} - {market_close} GMT")
        
        # CRITICAL DEBUG: Check if these hours make sense
        hour_diff = (datetime.combine(datetime.today(), market_close) - 
                    datetime.combine(datetime.today(), market_open)).total_seconds() / 3600
        logger.info(f"Market session length: {hour_diff:.1f} hours")
        
        if hour_diff > 10:
            logger.error(f"CRITICAL: Market session too long ({hour_diff:.1f} hours)!")
            logger.error("This will create excessive null slots and may cause data bleeding")
            logger.error("Danish market typically trades ~8 hours, not 10+")
            
        if market_open.hour < 6:
            logger.warning(f"SUSPICIOUS: Market open very early ({market_open}) - check timezone conversion")
            
        if market_close.hour > 18:
            logger.warning(f"SUSPICIOUS: Market close very late ({market_close}) - check timezone conversion")
        logger.info(f"Based on {len(daily_global_hours)} trading days with sufficient data")
        
        return market_open, market_close
    
    def _calculate_typical_time(self, times: List[time]) -> time:
        """Calculate the most typical time from a list of times."""
        if not times:
            return time(9, 0)  # Default fallback
        
        # Convert to minutes since midnight for easier calculation
        minutes_list = [t.hour * 60 + t.minute for t in times]
        
        # Use median as a robust estimate
        median_minutes = int(np.median(minutes_list))
        
        # Convert back to time
        hours = median_minutes // 60
        minutes = median_minutes % 60
        
        return time(hours, minutes)
    
    def create_complete_timeline(self, df: pl.DataFrame, ticker: str) -> pl.DataFrame:
        """
        Create a complete 1-minute timeline for market hours and fill missing values.
        
        Args:
            df: DataFrame with market data for a specific ticker
            ticker: Ticker symbol
            
        Returns:
            DataFrame with complete timeline and forward-filled prices
        """
        logger.info(f"Creating complete timeline for {ticker}")
        
        # Get market hours
        market_open, market_close = self.infer_market_hours(df, ticker)
        
        # Get date range from data
        ticker_data = df.filter(pl.col("ticker") == ticker)
        
        if ticker_data.is_empty():
            return df
        
        # Get min and max dates
        min_date = ticker_data.select(pl.col("datetime").dt.date().min()).item()
        max_date = ticker_data.select(pl.col("datetime").dt.date().max()).item()
        
        # Create complete timeline
        complete_timeline = self._generate_market_timeline(
            min_date, max_date, market_open, market_close
        )
        
        # Convert to Polars DataFrame
        timeline_df = pl.DataFrame({
            "datetime": complete_timeline,
            "ticker": [ticker] * len(complete_timeline)
        })
        
        # Join with actual data and forward fill
        complete_data = (timeline_df
            .join(ticker_data, on=["datetime", "ticker"], how="left")
            .sort("datetime")
        )
        
        # CRITICAL DEBUG: Check what we get BEFORE any forward filling
        if "high" in complete_data.columns:
            raw_values = complete_data.select("high").head(20).to_series().to_list()
            logger.debug(f"RAW joined data for {ticker} (first 20 high values): {raw_values}")
            
            # Count actual data points vs nulls
            total_rows = len(complete_data)
            null_count = complete_data.select(pl.col("high").is_null().sum()).item()
            data_percentage = ((total_rows - null_count) / total_rows) * 100
            logger.debug(f"Raw data density for {ticker}: {data_percentage:.1f}% ({total_rows - null_count}/{total_rows} non-null)")
            
            # Check if we have mostly nulls (indicates incomplete timeline)
            if data_percentage < 20:
                logger.warning(f"LOW DATA DENSITY for {ticker}: Only {data_percentage:.1f}% actual data")
                logger.warning("This timeline creation might be too aggressive - creating too many empty slots")
        
        # Forward fill price columns conservatively to avoid flat lines during market closure
        # Only fill small gaps in actual trading data, not extended periods
        price_columns = ["high", "low", "open", "close"]
        existing_price_cols = [col for col in price_columns if col in complete_data.columns]
        
        if existing_price_cols:
            # Add date column for grouping
            complete_data = complete_data.with_columns(
                pl.col("datetime").dt.date().alias("trading_date")
            )
            
            # CRITICAL FIX: The conservative forward-fill logic was flawed
            # The rolling sum approach was incorrectly filling nulls at market open
            # Let's use a much simpler approach: NO forward fill across day boundaries
            
            logger.debug(f"Applying day-boundary-aware forward fill for {ticker}")
            
            for col in existing_price_cols:
                # Get original column values before any modification
                original_values = complete_data.select(pl.col(col)).head(20).to_series().to_list()
                logger.debug(f"Original {col} values (first 20): {original_values}")
                
                # Simple rule: Only forward fill within very small gaps (1-2 minutes max)
                # This prevents any bleeding across market sessions
                
                # Create a marker for actual data points (not nulls)
                complete_data = complete_data.with_columns([
                    pl.col(col).is_not_null().alias(f"{col}_has_data")
                ])
                
                # Use a very conservative approach: only fill single null gaps
                # that are surrounded by actual data within the same day
                complete_data = complete_data.with_columns([
                    # Check if previous and next values exist (within same day)
                    pl.col(f"{col}_has_data").shift(1).over("trading_date").alias(f"{col}_prev_has_data"),
                    pl.col(f"{col}_has_data").shift(-1).over("trading_date").alias(f"{col}_next_has_data")
                ])
                
                # Only fill nulls that have data before AND after within the same day
                complete_data = complete_data.with_columns([
                    pl.when(
                        pl.col(col).is_null() & 
                        pl.col(f"{col}_prev_has_data") & 
                        pl.col(f"{col}_next_has_data")
                    )
                    .then(pl.col(col).fill_null(strategy="forward").over("trading_date"))
                    .otherwise(pl.col(col))
                    .alias(col)
                ])
                
                # Clean up temporary columns
                complete_data = complete_data.drop([
                    f"{col}_has_data", 
                    f"{col}_prev_has_data", 
                    f"{col}_next_has_data"
                ])
                
                # DEBUG: Check what happened
                filled_values = complete_data.select(pl.col(col)).head(20).to_series().to_list()
                logger.debug(f"After fill {col} values (first 20): {filled_values}")
                
                # Check if we accidentally filled market open nulls
                market_open_nulls = sum(1 for v in filled_values[:5] if v is None)
                if market_open_nulls < 3:  # Expect at least some nulls at market open
                    logger.warning(f"SUSPICIOUS: Only {market_open_nulls} nulls in first 5 market open values")
                    logger.warning("This might indicate data bleeding across market boundaries!")
                
                # More detailed analysis of what changed
                original_nulls = sum(1 for v in original_values[:20] if v is None)
                filled_nulls = sum(1 for v in filled_values[:20] if v is None)
                logger.debug(f"Fill operation changed nulls from {original_nulls}/20 to {filled_nulls}/20")
                
                if filled_nulls < original_nulls - 5:  # Filled too many nulls
                    logger.error(f"CRITICAL: Forward fill too aggressive for {col}!")
                    logger.error(f"Filled {original_nulls - filled_nulls} nulls that should have remained null")
                    logger.error("This is likely causing the data bleeding issue!")
                    
                    # Show which positions got filled
                    for i in range(min(20, len(original_values), len(filled_values))):
                        if original_values[i] is None and filled_values[i] is not None:
                            logger.error(f"Position {i}: NULL → {filled_values[i]:.2f} (SHOULD STAY NULL!)")
            
            # Remove temporary date column
            complete_data = complete_data.drop("trading_date")
        
        # Log information about null patterns for debugging
        if existing_price_cols:
            sample_col = existing_price_cols[0]
            null_count = complete_data.select(pl.col(sample_col).is_null().sum()).item()
            total_count = len(complete_data)
            logger.debug(f"Timeline created for {ticker}: {null_count}/{total_count} nulls preserved in {sample_col} (prevents flat lines)")
        
        logger.debug(f"Timeline created with proper day boundaries for {ticker}")
        
        return complete_data
    
    def _generate_market_timeline(self, start_date: datetime.date, end_date: datetime.date,
                                market_open: time, market_close: time) -> List[datetime]:
        """
        Generate complete 1-minute timeline for market hours between dates.
        
        Args:
            start_date: Start date
            end_date: End date
            market_open: Market opening time
            market_close: Market closing time
            
        Returns:
            List of datetime objects for each minute during market hours
        """
        timeline = []
        current_date = start_date
        
        while current_date <= end_date:
            # Skip weekends (Saturday=5, Sunday=6)
            if current_date.weekday() < 5:  # Monday=0, Friday=4
                
                # Create datetime objects for each minute during market hours
                current_datetime = datetime.combine(current_date, market_open)
                end_datetime = datetime.combine(current_date, market_close)
                
                while current_datetime <= end_datetime:
                    timeline.append(current_datetime)
                    current_datetime += timedelta(minutes=1)
            
            current_date += timedelta(days=1)
        
        return timeline
    
    def standardize_all_instruments(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Standardize datetime and create complete timeline for all instruments.
        
        Args:
            df: DataFrame with multiple instruments
            
        Returns:
            DataFrame with standardized datetime and complete timelines
        """
        import time
        start_time = time.time()
        
        logger.info("=== DATETIME STANDARDIZATION PROCESS ===")
        logger.info(f"Input data: {len(df)} rows, {len(df.columns)} columns")
        
        # First standardize datetime to GMT
        logger.info("Sub-step 1/4: Converting datetime to GMT...")
        step_start = time.time()
        df = self.standardize_to_gmt(df)
        logger.info(f"✓ GMT conversion completed in {time.time() - step_start:.2f}s")
        
        # Get unique tickers
        logger.info("Sub-step 2/4: Analyzing instruments...")
        step_start = time.time()
        tickers = df.select("ticker").unique().to_series().to_list()
        logger.info(f"✓ Found {len(tickers)} unique instruments in {time.time() - step_start:.2f}s")
        
        # Infer global market hours (once for all instruments)
        logger.info("Sub-step 3/4: Inferring global market hours...")
        step_start = time.time()
        market_open, market_close = self.infer_market_hours(df)
        logger.info(f"✓ Market hours inferred in {time.time() - step_start:.2f}s: {market_open} - {market_close}")
        
        # Process each ticker separately
        logger.info("Sub-step 4/4: Creating complete timelines for all instruments...")
        step_start = time.time()
        standardized_dfs = []
        
        for i, ticker in enumerate(tickers, 1):
            ticker_start = time.time()
            logger.debug(f"Processing {i}/{len(tickers)}: {ticker}")
            ticker_df = self.create_complete_timeline(df, ticker)
            standardized_dfs.append(ticker_df)
            
            if i % 10 == 0:  # Log progress every 10 instruments
                elapsed = time.time() - step_start
                avg_time = elapsed / i
                remaining = (len(tickers) - i) * avg_time
                logger.info(f"Progress: {i}/{len(tickers)} instruments processed ({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")
        
        logger.info(f"✓ All timeline creation completed in {time.time() - step_start:.2f}s")
        
        # Combine all tickers
        logger.info("Combining all instrument data...")
        combine_start = time.time()
        result_df = pl.concat(standardized_dfs)
        logger.info(f"✓ Data concatenation completed in {time.time() - combine_start:.2f}s")
        
        # Sort by ticker and datetime
        logger.info("Final sorting...")
        sort_start = time.time()
        result_df = result_df.sort(["ticker", "datetime"])
        logger.info(f"✓ Sorting completed in {time.time() - sort_start:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"=== DATETIME STANDARDIZATION COMPLETED in {total_time:.2f}s ===")
        logger.info(f"Final shape: {result_df.shape} (from {len(df)} to {len(result_df)} rows)")
        
        return result_df