"""Market data collection module for Saxo Bank API."""

import requests
import polars as pl
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from .auth import SaxoAuthenticator
from ..config.settings import settings

logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Collects market data from Saxo Bank API."""
    
    def __init__(self):
        self.authenticator = SaxoAuthenticator()
        self.base_url = settings.saxo_base_url
        self.delay = settings.api_delay
        self.interval_days = settings.data_interval_days
        
    def get_danish_stocks(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Retrieve list of Danish stocks from Saxo Bank API.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing stock data
            
        Raises:
            Exception: If API request fails
        """
        url = f"{self.base_url}/ref/v1/instruments"
        headers = self.authenticator.get_auth_headers()
        
        params = {
            "AssetTypes": "Stock",
            "ExchangeId": settings.default_exchange,
            "$top": limit,
            "IncludeNonTradable": "false"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Get stocks failed: {response.status_code} - {response.text}")
                raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            raise Exception(f"Request exception occurred: {str(e)}")
    
    def get_historical_data(self, instrument_uic: int, end_date: str) -> Dict[str, Any]:
        """
        Retrieve historical price data for a specific instrument.
        
        Args:
            instrument_uic: Unique instrument code
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary containing historical data
            
        Raises:
            Exception: If API request fails
        """
        url = f"{self.base_url}/chart/v3/charts"
        headers = self.authenticator.get_auth_headers()
        
        params = {
            "Uic": instrument_uic,
            "AssetType": "Stock",
            "Horizon": 1,  # 1-minute intervals
            "Time": f"{end_date}T00:00:00Z",
            "Mode": "UpTo",
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Get historical data failed: {response.status_code} - {response.text}")
                raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception: {e}")
            raise Exception(f"Request exception occurred: {str(e)}")
    
    def generate_date_intervals(self, start_date: str, end_date: str) -> List[str]:
        """
        Generate date intervals for data collection.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of date strings for intervals
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = []
        delta = timedelta(days=self.interval_days)
        
        current_date = start
        while current_date <= end:
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += delta
        
        if end_date not in dates:
            dates.append(end_date)
        
        return dates
    
    def collect_instrument_data(self, 
                              instrument_uic: int, 
                              symbol: str, 
                              name: str, 
                              start_date: str, 
                              end_date: str,
                              existing_data: Optional[pl.DataFrame] = None) -> List[Dict[str, Any]]:
        """
        Collect data for a single instrument across date range.
        
        Args:
            instrument_uic: Unique instrument code
            symbol: Stock symbol
            name: Stock name
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            existing_data: Optional existing data to check for duplicates
            
        Returns:
            List of data dictionaries
        """
        data_points = []
        date_intervals = self.generate_date_intervals(start_date, end_date)
        
        # Create set of existing dates if data provided
        existing_dates = set()
        if existing_data is not None and not existing_data.is_empty():
            existing_dates = set(
                existing_data
                .filter(pl.col("uic") == instrument_uic)
                .with_columns(pl.col("datetime").str.slice(0, 10).alias("date"))
                .select("date")
                .to_series()
                .to_list()
            )
        
        logger.info(f"Collecting data for {name} ({symbol})")
        
        for date in date_intervals:
            # Skip if we already have data for this date
            if date in existing_dates:
                logger.debug(f"Skipping {date} for {symbol} - already exists")
                continue
            
            logger.debug(f"Collecting data for {symbol} on {date}")
            
            try:
                hist_data = self.get_historical_data(instrument_uic, date)
                
                if "Data" in hist_data:
                    for data_point in hist_data["Data"]:
                        data_points.append({
                            "datetime": data_point["Time"],
                            "high": data_point["High"],
                            "low": data_point["Low"],
                            "open": data_point["Open"],
                            "close": data_point["Close"],
                            "uic": instrument_uic,
                            "name": name,
                            "ticker": symbol
                        })
                
                # Rate limiting
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol} on {date}: {e}")
                continue
        
        return data_points
    
    def collect_market_data(self, 
                          start_date: str, 
                          end_date: str,
                          output_file: str = "danish_stocks_1m.csv",
                          update_existing: bool = False) -> None:
        """
        Collect market data for all Danish stocks.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_file: Output CSV filename
            update_existing: Whether to update existing data
        """
        logger.info(f"Starting data collection from {start_date} to {end_date}")
        
        # Ensure we have a valid access token
        if not self.authenticator.ensure_valid_token():
            raise Exception("Failed to obtain valid access token. Please check your credentials.")
        
        # Get list of Danish stocks
        stocks_data = self.get_danish_stocks()
        
        if "Data" not in stocks_data:
            raise Exception("No stock data returned from API")
        
        stocks = stocks_data["Data"]
        logger.info(f"Found {len(stocks)} Danish stocks")
        
        # Load existing data if updating
        existing_data = None
        output_path = settings.get_raw_data_file(output_file)
        
        if update_existing and output_path.exists():
            logger.info("Loading existing data for update")
            existing_data = pl.read_csv(output_path)
        
        all_data = []
        last_token_check = time.time()
        token_check_interval = 400  # Check every hour (3600 seconds)
        
        for stock in stocks:
            uic = stock["Identifier"]
            symbol = stock["Symbol"]
            name = stock["Description"]
            
            # Check token validity every hour
            current_time = time.time()
            #if current_time - last_token_check > token_check_interval:
            #logger.info("Checking token validity during long collection process")
            if not self.authenticator.ensure_valid_token():
                raise Exception("Failed to maintain valid access token during collection")
            last_token_check = current_time
            
            try:
                instrument_data = self.collect_instrument_data(
                    uic, symbol, name, start_date, end_date, existing_data
                )
                all_data.extend(instrument_data)
                
            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}: {e}")
                continue
        
        # Create DataFrame and save
        if all_data:
            new_df = pl.DataFrame(all_data)
            
            if update_existing and existing_data is not None:
                # Combine with existing data
                final_df = pl.concat([existing_data, new_df])
            else:
                final_df = new_df
            
            # Remove duplicates and sort
            final_df = (final_df
                       .unique(["uic", "datetime"])
                       .sort(["uic", "datetime"]))
            
            # Save to file
            final_df.write_csv(output_path)
            logger.info(f"Saved {len(final_df)} data points to {output_path}")
            
        else:
            logger.warning("No new data collected")
    
    def get_instrument_list(self) -> pl.DataFrame:
        """
        Get list of available instruments as DataFrame.
        
        Returns:
            DataFrame with instrument information
        """
        stocks_data = self.get_danish_stocks()
        
        if "Data" not in stocks_data:
            raise Exception("No stock data returned from API")
        
        instruments = []
        for stock in stocks_data["Data"]:
            instruments.append({
                "uic": stock["Identifier"],
                "symbol": stock["Symbol"],
                "name": stock["Description"],
                "exchange": stock.get("ExchangeId", ""),
                "currency": stock.get("CurrencyCode", ""),
                "asset_type": stock.get("AssetType", "")
            })
        
        return pl.DataFrame(instruments)
    
    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """
        Validate date range format and logic.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            True if valid, False otherwise
        """
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            return start <= end
        except ValueError:
            return False