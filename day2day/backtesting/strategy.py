"""Trading strategy implementation."""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..models.bootstrap import BootstrapEstimator
from ..models.training import ModelTrainer
from ..config.settings import settings


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    direction: str  # 'long' or 'short'
    status: str  # 'open', 'closed'
    pnl: float = 0.0
    fees: float = 0.0


class TradingStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, 
                 initial_capital: float,
                 max_positions: int = 1,
                 exchange_fee: float = 0.0008,
                 fixed_cost: float = 20.0):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.exchange_fee = exchange_fee
        self.fixed_cost = fixed_cost
        self.current_capital = initial_capital
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        
    @abstractmethod
    def should_enter_position(self, 
                            current_price: float,
                            prediction: float,
                            probability: float,
                            timestamp: pd.Timestamp) -> Tuple[bool, float]:
        """
        Determine if should enter a position.
        
        Args:
            current_price: Current market price
            prediction: Model prediction
            probability: Bootstrap probability
            timestamp: Current timestamp
            
        Returns:
            Tuple of (should_enter, position_size)
        """
        pass
    
    @abstractmethod
    def should_exit_position(self, 
                           trade: Trade,
                           current_price: float,
                           timestamp: pd.Timestamp) -> bool:
        """
        Determine if should exit a position.
        
        Args:
            trade: Current trade
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Whether to exit the position
        """
        pass
    
    def calculate_position_size(self, 
                              current_price: float,
                              available_capital: float) -> float:
        """
        Calculate position size based on available capital.
        
        Args:
            current_price: Current market price
            available_capital: Available capital
            
        Returns:
            Position size in shares
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Distribute capital across max positions
        position_capital = available_capital / self.max_positions
        
        # Account for fees
        effective_capital = position_capital - self.fixed_cost
        
        if effective_capital <= 0:
            return 0.0
        
        # Calculate shares (accounting for entry fee)
        shares = effective_capital / (current_price * (1 + self.exchange_fee))
        
        # Debug position sizing for first few trades
        debug_counter = getattr(self, '_position_size_counter', 0)
        self._position_size_counter = debug_counter + 1
        
        if debug_counter < 10:
            logger.info(f"🔢 POSITION SIZE CALCULATION #{debug_counter + 1}:")
            logger.info(f"  Current Price: ${current_price:.6f}")
            logger.info(f"  Available Capital: ${available_capital:.2f}")
            logger.info(f"  Position Capital: ${position_capital:.2f}")
            logger.info(f"  Effective Capital: ${effective_capital:.2f}")
            logger.info(f"  Calculated Shares: {shares:.2f}")
            logger.info(f"  Position Value: ${shares * current_price:.2f}")
            
            # Check for unrealistic position sizes
            if shares > 100000:  # More than 100k shares
                logger.error(f"🚨 MASSIVE POSITION SIZE DETECTED!")
                logger.error(f"   Shares: {shares:,.0f}")
                logger.error(f"   This suggests extremely low prices or calculation bug!")
                logger.error(f"   Price may be in wrong units (e.g., percentage instead of dollars)")
        
        return max(0.0, shares)
    
    def enter_position(self, 
                      current_price: float,
                      quantity: float,
                      timestamp: pd.Timestamp,
                      direction: str = 'long') -> Trade:
        """
        Enter a new position.
        
        Args:
            current_price: Entry price
            quantity: Position size
            timestamp: Entry timestamp
            direction: Position direction
            
        Returns:
            New trade object
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Calculate fees
        position_value = current_price * quantity
        entry_fee = position_value * self.exchange_fee + self.fixed_cost
        
        # Debug capital tracking
        capital_before = self.current_capital
        
        # Create trade
        trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            entry_price=current_price,
            exit_price=None,
            quantity=quantity,
            direction=direction,
            status='open',
            fees=entry_fee
        )
        
        # Update capital
        self.current_capital -= position_value + entry_fee
        
        # Debug logging for first few trades
        if len(self.closed_trades) + len(self.open_trades) <= 10:
            logger.info(f"📈 ENTER POSITION #{len(self.closed_trades) + len(self.open_trades)}:")
            logger.info(f"  Price: ${current_price:.4f}, Quantity: {quantity:.2f}")
            logger.info(f"  Position Value: ${position_value:.2f}")
            logger.info(f"  Entry Fee: ${entry_fee:.2f}")
            logger.info(f"  Capital: ${capital_before:.2f} → ${self.current_capital:.2f}")
            logger.info(f"  Capital Change: ${capital_before - self.current_capital:.2f}")
        
        # Add to open trades
        self.open_trades.append(trade)
        
        return trade
    
    def exit_position(self, 
                     trade: Trade,
                     current_price: float,
                     timestamp: pd.Timestamp) -> Trade:
        """
        Exit an existing position.
        
        Args:
            trade: Trade to exit
            current_price: Exit price
            timestamp: Exit timestamp
            
        Returns:
            Updated trade object
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Calculate exit fees
        position_value = current_price * trade.quantity
        exit_fee = position_value * self.exchange_fee + self.fixed_cost
        
        # Debug capital tracking
        capital_before = self.current_capital
        
        # Update trade
        trade.exit_time = timestamp
        trade.exit_price = current_price
        trade.status = 'closed'
        trade.fees += exit_fee
        
        # Calculate PnL
        if trade.direction == 'long':
            gross_pnl = (current_price - trade.entry_price) * trade.quantity
        else:
            gross_pnl = (trade.entry_price - current_price) * trade.quantity
        
        trade.pnl = gross_pnl - trade.fees
        
        # Update capital
        self.current_capital += position_value - exit_fee
        
        # Debug logging for first few trades
        if len(self.closed_trades) <= 10:
            logger.info(f"📉 EXIT POSITION #{len(self.closed_trades) + 1}:")
            logger.info(f"  Entry: ${trade.entry_price:.4f} → Exit: ${current_price:.4f}")
            logger.info(f"  Quantity: {trade.quantity:.2f}")
            logger.info(f"  Position Value: ${position_value:.2f}")
            logger.info(f"  Exit Fee: ${exit_fee:.2f}")
            logger.info(f"  Total Fees: ${trade.fees:.2f}")
            logger.info(f"  Gross PnL: ${gross_pnl:.2f}")
            logger.info(f"  Net PnL: ${trade.pnl:.2f}")
            logger.info(f"  Capital: ${capital_before:.2f} → ${self.current_capital:.2f}")
            logger.info(f"  Capital Change: ${self.current_capital - capital_before:.2f}")
        
        # Move to closed trades
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)
        
        return trade
    
    def force_close_all_positions(self, 
                                current_price: float,
                                timestamp: pd.Timestamp) -> List[Trade]:
        """
        Force close all open positions (end of day).
        
        Args:
            current_price: Current market price
            timestamp: Close timestamp
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for trade in self.open_trades.copy():
            closed_trade = self.exit_position(trade, current_price, timestamp)
            closed_trades.append(closed_trade)
        
        return closed_trades
    
    def get_available_capital(self) -> float:
        """Get available capital for new positions."""
        return self.current_capital
    
    def get_total_value(self, current_price: float) -> float:
        """
        Get total portfolio value.
        
        Args:
            current_price: Current market price
            
        Returns:
            Total portfolio value
        """
        import logging
        logger = logging.getLogger(__name__)
        
        total_value = self.current_capital
        
        # Debug counter for portfolio value calculations
        debug_counter = getattr(self, '_portfolio_debug_counter', 0)
        self._portfolio_debug_counter = debug_counter + 1
        should_debug = debug_counter < 10 or debug_counter % 1000 == 0
        
        if should_debug:
            logger.info(f"💰 PORTFOLIO VALUE #{debug_counter}:")
            logger.info(f"  Current Price: ${current_price:.4f}")
            logger.info(f"  Cash Capital: ${self.current_capital:.2f}")
            logger.info(f"  Open Positions: {len(self.open_trades)}")
        
        # Add value of open positions
        total_position_value = 0
        for i, trade in enumerate(self.open_trades):
            if trade.direction == 'long':
                position_value = current_price * trade.quantity
            else:
                position_value = trade.entry_price * trade.quantity - (current_price - trade.entry_price) * trade.quantity
            
            total_position_value += position_value
            
            if should_debug:
                logger.info(f"  Position {i+1}: {trade.quantity:.2f} shares @ ${current_price:.4f} = ${position_value:.2f}")
        
        total_value += total_position_value
        
        if should_debug:
            logger.info(f"  Total Position Value: ${total_position_value:.2f}")
            logger.info(f"  TOTAL PORTFOLIO: ${total_value:.2f}")
            
            # Check for unrealistic values
            if total_value > self.initial_capital * 10:  # More than 10x initial capital
                logger.error(f"🚨 UNREALISTIC PORTFOLIO VALUE DETECTED!")
                logger.error(f"   Initial: ${self.initial_capital:.2f}")
                logger.error(f"   Current: ${total_value:.2f}")
                logger.error(f"   Multiplier: {total_value / self.initial_capital:.1f}x")
                logger.error(f"   This suggests a calculation bug!")
        
        return total_value


class ProbabilityBasedStrategy(TradingStrategy):
    """Strategy based on prediction probability thresholds."""
    
    def __init__(self,
                 initial_capital: float,
                 min_probability: float,
                 price_increase_threshold: float,
                 take_profit_threshold: float,
                 stop_loss_threshold: float,
                 threshold_is_fraction: bool = True,
                 **kwargs):
        super().__init__(initial_capital, **kwargs)
        self.min_probability = min_probability
        self.price_increase_threshold = price_increase_threshold
        self.take_profit_threshold = take_profit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.threshold_is_fraction = threshold_is_fraction
        
        # Log strategy configuration for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"📊 Strategy Configuration:")
        logger.info(f"  Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"  Min Probability: {min_probability:.1%}")
        logger.info(f"  Price Increase Threshold: {price_increase_threshold} {'(fraction)' if threshold_is_fraction else '(absolute)'}")
        logger.info(f"  Take Profit Threshold: {take_profit_threshold} {'(fraction)' if threshold_is_fraction else '(absolute)'}")
        logger.info(f"  Stop Loss Threshold: {stop_loss_threshold} {'(fraction)' if threshold_is_fraction else '(absolute)'}")
        logger.info(f"  Max Positions: {kwargs.get('max_positions', 1)}")
        logger.info(f"  Exchange Fee: {kwargs.get('exchange_fee', 0.0008):.4f}")
        logger.info(f"  Fixed Cost per Trade: ${kwargs.get('fixed_cost', 20.0):.2f}")
        
        if threshold_is_fraction:
            logger.info(f"  💡 Example thresholds for $100 stock:")
            logger.info(f"    Required price increase: ${100 * price_increase_threshold:.2f}")
            logger.info(f"    Take profit at: ${100 * (1 + take_profit_threshold):.2f}")
            logger.info(f"    Stop loss at: ${100 * (1 - stop_loss_threshold):.2f}")
    
    def should_enter_position(self, 
                            current_price: float,
                            prediction: float,
                            probability: float,
                            timestamp: pd.Timestamp) -> Tuple[bool, float]:
        """
        Enter position if probability exceeds threshold.
        
        Args:
            current_price: Current market price
            prediction: Model prediction
            probability: Bootstrap probability
            timestamp: Current timestamp
            
        Returns:
            Tuple of (should_enter, position_size)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Debug: Log every 500th evaluation to avoid spam (less verbose for speed)
        debug_counter = getattr(self, '_debug_counter', 0)
        self._debug_counter = debug_counter + 1
        should_debug = (debug_counter % 500 == 0) or debug_counter < 5
        
        if should_debug:
            logger.info(f"🔍 Entry evaluation #{debug_counter}:")
            logger.info(f"  Time: {timestamp}")
            logger.info(f"  Price: ${current_price:.4f}")
            logger.info(f"  Prediction: ${prediction:.4f}")
            logger.info(f"  Probability: {probability:.4f}")
            logger.info(f"  Open positions: {len(self.open_trades)}/{self.max_positions}")
        
        # Check if we have room for more positions
        if len(self.open_trades) >= self.max_positions:
            if should_debug:
                logger.info(f"  ❌ Max positions reached ({len(self.open_trades)}/{self.max_positions})")
            return False, 0.0
        
        # Check if probability meets threshold
        if probability < self.min_probability:
            if should_debug:
                logger.info(f"  ❌ Probability too low: {probability:.4f} < {self.min_probability}")
            return False, 0.0
        
        # Check if predicted increase meets threshold
        if self.threshold_is_fraction:
            required_increase = current_price * self.price_increase_threshold
        else:
            required_increase = self.price_increase_threshold
        
        predicted_gain = prediction - current_price
        
        if prediction < current_price + required_increase:
            if should_debug:
                logger.info(f"  ❌ Predicted gain too small:")
                logger.info(f"    Predicted gain: ${predicted_gain:.4f}")
                logger.info(f"    Required gain: ${required_increase:.4f}")
                logger.info(f"    Threshold: {self.price_increase_threshold} {'(fraction)' if self.threshold_is_fraction else '(absolute)'}")
            return False, 0.0
        
        # Calculate position size
        available_capital = self.get_available_capital()
        position_size = self.calculate_position_size(current_price, available_capital)
        
        if position_size <= 0:
            if should_debug:
                logger.info(f"  ❌ No capital available for position (capital: ${available_capital:.2f})")
            return False, 0.0
        
        # All checks passed - would enter position!
        if should_debug:
            logger.info(f"  ✅ WOULD ENTER POSITION!")
            logger.info(f"    Position size: {position_size:.2f} shares")
            logger.info(f"    Capital used: ${current_price * position_size:.2f}")
            logger.info(f"    Available capital: ${available_capital:.2f}")
        
        # Log every successful entry evaluation for first 50 attempts
        if not should_debug and debug_counter < 50:
            logger.info(f"🎯 Entry opportunity #{debug_counter}: Price=${current_price:.4f}, Pred=${prediction:.4f}, Prob={probability:.4f}")
        
        return position_size > 0, position_size
    
    def should_exit_position(self, 
                           trade: Trade,
                           current_price: float,
                           timestamp: pd.Timestamp) -> bool:
        """
        Exit position based on take profit or stop loss.
        
        Args:
            trade: Current trade
            current_price: Current market price
            timestamp: Current timestamp
            
        Returns:
            Whether to exit the position
        """
        if trade.direction == 'long':
            # Calculate profit/loss
            pnl_per_share = current_price - trade.entry_price
            
            # Take profit
            if self.threshold_is_fraction:
                take_profit_price = trade.entry_price * (1 + self.take_profit_threshold)
            else:
                take_profit_price = trade.entry_price + self.take_profit_threshold
            
            if current_price >= take_profit_price:
                return True
            
            # Stop loss
            if self.threshold_is_fraction:
                stop_loss_price = trade.entry_price * (1 - self.stop_loss_threshold)
            else:
                stop_loss_price = trade.entry_price - self.stop_loss_threshold
            
            if current_price <= stop_loss_price:
                return True
        
        return False
    
    def is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within market hours."""
        time = timestamp.time()
        return datetime.time(9, 30) <= time <= datetime.time(16, 0)
    
    def is_end_of_day(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is end of trading day."""
        return timestamp.time() >= datetime.time(15, 55)  # 5 minutes before close


class ModelBasedStrategy(ProbabilityBasedStrategy):
    """Strategy that uses model predictions and bootstrap probabilities."""
    
    def __init__(self,
                 training_data_title: str,
                 target_instrument: str,
                 model_name: str,
                 **kwargs):
        super().__init__(**kwargs)
        self.training_data_title = training_data_title
        self.target_instrument = target_instrument
        self.model_name = model_name
        
        # Load model and bootstrap results
        self.trainer = ModelTrainer()
        self.bootstrap_estimator = BootstrapEstimator()
        
        # Load model
        self.models = self.trainer.load_model_suite(training_data_title, target_instrument)
        self.model = self.models[model_name]
        
        # Determine model type for bootstrap loading
        # Extract base model type from model name (e.g., "xgboost_default" -> "xgboost")
        if hasattr(self.model, 'model_type'):
            model_type_for_bootstrap = self.model.model_type
        else:
            # Fallback: extract from model name by taking first part before underscore
            model_type_for_bootstrap = model_name.split('_')[0]
        
        # Load bootstrap results using the base model type
        self.bootstrap_results = self.bootstrap_estimator.load_bootstrap_results(
            training_data_title, target_instrument, model_type_for_bootstrap
        )
    
    def get_prediction_and_probability(self, 
                                     features: pd.DataFrame,
                                     current_price: float) -> Tuple[float, float]:
        """
        Get model prediction and bootstrap probability.
        
        Args:
            features: Feature vector
            current_price: Current market price
            
        Returns:
            Tuple of (prediction, probability)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Get model prediction
        prediction = self.model.predict(features)[0]
        
        # Calculate threshold for probability
        if self.threshold_is_fraction:
            threshold = current_price * (1 + self.price_increase_threshold)
        else:
            threshold = current_price + self.price_increase_threshold
        
        # ISSUE: Bootstrap predictions were made on training/test data, not current features
        # We need to make bootstrap predictions on current features instead
        
        # Get bootstrap predictions for current features
        bootstrap_predictions = self.bootstrap_results['bootstrap_predictions']
        
        debug_counter = getattr(self, '_prob_debug_counter', 0)
        self._prob_debug_counter = debug_counter + 1
        should_debug = debug_counter < 5 or debug_counter % 500 == 0
        
        if should_debug:
            logger.info(f"🎲 Bootstrap probability calculation #{debug_counter}:")
            logger.info(f"  Current prediction: ${prediction:.6f}")
            logger.info(f"  Current price: ${current_price:.6f}")
            logger.info(f"  Threshold: ${threshold:.6f}")
            logger.info(f"  Bootstrap predictions shape: {bootstrap_predictions.shape}")
        
        # PROBLEM: Bootstrap predictions are for test set, not for current features!
        # For real-time prediction, we would need to run bootstrap models on current features
        # For now, let's use a workaround: estimate probability based on prediction vs threshold
        
        # Workaround: Use simple probability estimate based on how far prediction exceeds threshold
        if prediction > threshold:
            # Simple heuristic: probability based on how much prediction exceeds threshold
            excess = prediction - threshold
            relative_excess = excess / current_price if current_price > 0 else 0
            # Convert to probability (sigmoid-like function)
            probability = min(0.95, max(0.05, 0.5 + relative_excess * 10))
            
            if should_debug:
                logger.info(f"  ✅ Prediction exceeds threshold by ${excess:.6f} ({relative_excess:.1%})")
                logger.info(f"  Estimated probability: {probability:.4f}")
        else:
            # Prediction below threshold
            shortfall = threshold - prediction
            relative_shortfall = shortfall / current_price if current_price > 0 else 0
            probability = max(0.01, 0.5 - relative_shortfall * 10)
            
            if should_debug:
                logger.info(f"  ❌ Prediction below threshold by ${shortfall:.6f} ({relative_shortfall:.1%})")
                logger.info(f"  Estimated probability: {probability:.4f}")
        
        # TODO: For proper bootstrap probability, we would need to:
        # 1. Load all bootstrap models
        # 2. Make predictions on current features with each model
        # 3. Calculate probability that predictions exceed threshold
        # This is computationally expensive so using heuristic for now
        
        if should_debug:
            logger.warning(f"  ⚠️  Using heuristic probability estimation (not true bootstrap)")
            logger.warning(f"     For accurate probabilities, bootstrap models would need to predict on current features")
        
        return prediction, probability