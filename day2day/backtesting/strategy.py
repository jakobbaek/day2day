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
        # Distribute capital across max positions
        position_capital = available_capital / self.max_positions
        
        # Account for fees
        effective_capital = position_capital - self.fixed_cost
        
        if effective_capital <= 0:
            return 0.0
        
        # Calculate shares (accounting for entry fee)
        shares = effective_capital / (current_price * (1 + self.exchange_fee))
        
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
        # Calculate fees
        position_value = current_price * quantity
        entry_fee = position_value * self.exchange_fee + self.fixed_cost
        
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
        # Calculate exit fees
        position_value = current_price * trade.quantity
        exit_fee = position_value * self.exchange_fee + self.fixed_cost
        
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
        total_value = self.current_capital
        
        # Add value of open positions
        for trade in self.open_trades:
            if trade.direction == 'long':
                position_value = current_price * trade.quantity
            else:
                position_value = trade.entry_price * trade.quantity - (current_price - trade.entry_price) * trade.quantity
            
            total_value += position_value
        
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
        # Check if we have room for more positions
        if len(self.open_trades) >= self.max_positions:
            return False, 0.0
        
        # Check if probability meets threshold
        if probability < self.min_probability:
            return False, 0.0
        
        # Check if predicted increase meets threshold
        if self.threshold_is_fraction:
            required_increase = current_price * self.price_increase_threshold
        else:
            required_increase = self.price_increase_threshold
        
        if prediction < current_price + required_increase:
            return False, 0.0
        
        # Calculate position size
        available_capital = self.get_available_capital()
        position_size = self.calculate_position_size(current_price, available_capital)
        
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
        # Get model prediction
        prediction = self.model.predict(features)[0]
        
        # Calculate threshold for probability
        if self.threshold_is_fraction:
            threshold = current_price * (1 + self.price_increase_threshold)
        else:
            threshold = current_price + self.price_increase_threshold
        
        # Get probability from bootstrap results
        bootstrap_predictions = self.bootstrap_results['bootstrap_predictions']
        probabilities = self.bootstrap_estimator.estimate_prediction_probability(
            bootstrap_predictions, threshold, 'above'
        )
        
        # Use latest probability (assuming features correspond to latest time)
        probability = probabilities[-1] if len(probabilities) > 0 else 0.0
        
        return prediction, probability