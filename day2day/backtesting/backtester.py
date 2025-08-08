"""Backtesting engine implementation."""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging
from .strategy import TradingStrategy, Trade
from .results import BacktestResults
from ..config.settings import settings

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self):
        self.settings = settings
        
    def load_test_data(self, 
                      training_data_title: str,
                      target_instrument: str) -> pd.DataFrame:
        """
        Load test data for backtesting.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            
        Returns:
            Test data DataFrame
        """
        suite_name = f"{training_data_title}_{target_instrument}"
        suite_dir = settings.models_path / suite_name
        test_file = suite_dir / "test_data.csv"
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        
        df = pd.read_csv(test_file)
        
        # Convert numeric columns that were incorrectly saved as objects
        exclude_from_conversion = ['datetime', 'trading_eligible']
        for col in df.columns:
            if col not in exclude_from_conversion and df[col].dtype == 'object':
                # Try to convert to numeric, keep as object if conversion fails
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.debug(f"Converted column {col} from object to numeric")
                except:
                    logger.warning(f"Could not convert column {col} to numeric")
        
        # Add datetime index if not present
        if 'datetime' not in df.columns:
            # Create synthetic datetime index
            df['datetime'] = pd.date_range(
                start='2023-01-01 09:30:00',
                periods=len(df),
                freq='5min'
            )
        else:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        return df
    
    def run_backtest(self,
                    strategy: TradingStrategy,
                    test_data: pd.DataFrame,
                    price_column: str = 'target') -> BacktestResults:
        """
        Run backtest with given strategy.
        
        Args:
            strategy: Trading strategy to test
            test_data: Test data
            price_column: Column name for prices
            
        Returns:
            Backtest results
        """
        logger.info(f"Running backtest with {len(test_data)} data points")
        
        # Ensure data is sorted by datetime
        test_data = test_data.sort_values('datetime').reset_index(drop=True)
        
        # Track portfolio values
        portfolio_values = []
        timestamps = []
        
        # Main backtesting loop
        for i, row in test_data.iterrows():
            timestamp = pd.to_datetime(row['datetime'])
            current_price = row[price_column]
            
            # Skip if price is invalid
            if pd.isna(current_price) or current_price <= 0:
                continue
            
            # Check for end of day - close all positions
            if self._is_end_of_day(timestamp):
                strategy.force_close_all_positions(current_price, timestamp)
            
            # Check exit conditions for open positions
            for trade in strategy.open_trades.copy():
                if strategy.should_exit_position(trade, current_price, timestamp):
                    strategy.exit_position(trade, current_price, timestamp)
            
            # Check entry conditions (only during market hours)
            if self._is_market_hours(timestamp) and not self._is_end_of_day(timestamp):
                should_enter, position_size = self._get_entry_signal(
                    strategy, row, current_price, timestamp
                )
                
                if should_enter and position_size > 0:
                    strategy.enter_position(current_price, position_size, timestamp)
            
            # Record portfolio value
            portfolio_value = strategy.get_total_value(current_price)
            portfolio_values.append(portfolio_value)
            timestamps.append(timestamp)
        
        # Close any remaining positions at the end
        if strategy.open_trades:
            final_price = test_data[price_column].iloc[-1]
            final_timestamp = pd.to_datetime(test_data['datetime'].iloc[-1])
            strategy.force_close_all_positions(final_price, final_timestamp)
        
        # Create results
        results = BacktestResults(
            strategy=strategy,
            portfolio_values=portfolio_values,
            timestamps=timestamps,
            initial_capital=strategy.initial_capital,
            final_capital=strategy.current_capital,
            trades=strategy.closed_trades
        )
        
        return results
    
    def _get_entry_signal(self,
                         strategy: TradingStrategy,
                         row: pd.Series,
                         current_price: float,
                         timestamp: pd.Timestamp) -> Tuple[bool, float]:
        """
        Get entry signal from strategy.
        
        Args:
            strategy: Trading strategy
            row: Current data row
            current_price: Current price
            timestamp: Current timestamp
            
        Returns:
            Tuple of (should_enter, position_size)
        """
        # For model-based strategies, we need to provide prediction and probability
        if hasattr(strategy, 'get_prediction_and_probability'):
            # Prepare features (exclude target, datetime, and trading_eligible)
            exclude_cols = ['target', 'datetime', 'trading_eligible']
            feature_cols = [col for col in row.index if col not in exclude_cols]
            features = row[feature_cols].to_frame().T
            
            # Convert object columns to numeric, replacing non-numeric with NaN
            for col in features.columns:
                if features[col].dtype == 'object':
                    features[col] = pd.to_numeric(features[col], errors='coerce')
            
            try:
                prediction, probability = strategy.get_prediction_and_probability(
                    features, current_price
                )
                
                return strategy.should_enter_position(
                    current_price, prediction, probability, timestamp
                )
            except Exception as e:
                logger.error(f"Error getting prediction: {e}")
                return False, 0.0
        else:
            # For simple strategies, use dummy values
            return strategy.should_enter_position(
                current_price, current_price * 1.01, 0.5, timestamp
            )
    
    def _is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within market hours."""
        time = timestamp.time()
        return datetime.time(9, 30) <= time <= datetime.time(16, 0)
    
    def _is_end_of_day(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is end of trading day."""
        return timestamp.time() >= datetime.time(15, 55)
    
    def run_parameter_sweep(self,
                          strategy_class: type,
                          base_params: Dict[str, Any],
                          param_ranges: Dict[str, List[Any]],
                          test_data: pd.DataFrame,
                          price_column: str = 'target') -> List[Dict[str, Any]]:
        """
        Run parameter sweep for strategy optimization.
        
        Args:
            strategy_class: Strategy class to test
            base_params: Base parameters for strategy
            param_ranges: Parameter ranges to test
            test_data: Test data
            price_column: Price column name
            
        Returns:
            List of results for each parameter combination
        """
        import itertools
        
        # Generate all parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        results = []
        
        for param_combo in itertools.product(*param_values):
            # Create parameter dict
            params = base_params.copy()
            for name, value in zip(param_names, param_combo):
                params[name] = value
            
            try:
                # Create strategy
                strategy = strategy_class(**params)
                
                # Run backtest
                backtest_results = self.run_backtest(strategy, test_data, price_column)
                
                # Store results
                result = {
                    'parameters': params,
                    'total_return': backtest_results.total_return,
                    'sharpe_ratio': backtest_results.sharpe_ratio,
                    'max_drawdown': backtest_results.max_drawdown,
                    'num_trades': backtest_results.num_trades,
                    'win_rate': backtest_results.win_rate,
                    'profit_factor': backtest_results.profit_factor
                }
                
                results.append(result)
                
                logger.info(f"Parameter combo: {params} -> Return: {backtest_results.total_return:.2%}")
                
            except Exception as e:
                logger.error(f"Error with parameters {params}: {e}")
                continue
        
        # Sort by total return
        results.sort(key=lambda x: x['total_return'], reverse=True)
        
        return results
    
    def save_backtest_results(self,
                            results: BacktestResults,
                            training_data_title: str,
                            target_instrument: str,
                            strategy_name: str) -> None:
        """
        Save backtest results.
        
        Args:
            results: Backtest results
            training_data_title: Title of training data
            target_instrument: Target instrument
            strategy_name: Name of strategy
        """
        # Create output directory
        output_dir = settings.backtests_path / f"{training_data_title}_{target_instrument}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        results.save_to_file(output_dir / f"{strategy_name}_backtest.json")
        
        # Save trades
        if results.trades:
            trades_df = pd.DataFrame([{
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'direction': trade.direction,
                'pnl': trade.pnl,
                'fees': trade.fees
            } for trade in results.trades])
            
            trades_df.to_csv(output_dir / f"{strategy_name}_trades.csv", index=False)
        
        # Save portfolio values
        portfolio_df = pd.DataFrame({
            'timestamp': results.timestamps,
            'portfolio_value': results.portfolio_values
        })
        
        portfolio_df.to_csv(output_dir / f"{strategy_name}_portfolio.csv", index=False)
        
        logger.info(f"Saved backtest results to {output_dir}")
    
    def compare_strategies(self,
                         strategies: List[TradingStrategy],
                         strategy_names: List[str],
                         test_data: pd.DataFrame,
                         price_column: str = 'target') -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategies: List of strategies to compare
            strategy_names: Names of strategies
            test_data: Test data
            price_column: Price column name
            
        Returns:
            Comparison DataFrame
        """
        results = []
        
        for strategy, name in zip(strategies, strategy_names):
            # Run backtest
            backtest_results = self.run_backtest(strategy, test_data, price_column)
            
            # Collect metrics
            results.append({
                'strategy': name,
                'total_return': backtest_results.total_return,
                'annualized_return': backtest_results.annualized_return,
                'volatility': backtest_results.volatility,
                'sharpe_ratio': backtest_results.sharpe_ratio,
                'max_drawdown': backtest_results.max_drawdown,
                'num_trades': backtest_results.num_trades,
                'win_rate': backtest_results.win_rate,
                'profit_factor': backtest_results.profit_factor,
                'avg_trade_duration': backtest_results.avg_trade_duration
            })
        
        return pd.DataFrame(results)