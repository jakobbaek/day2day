"""Backtesting results analysis and visualization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from .strategy import Trade
import logging

logger = logging.getLogger(__name__)


class BacktestResults:
    """Container for backtesting results and analysis."""
    
    def __init__(self,
                 strategy,
                 portfolio_values: List[float],
                 timestamps: List[pd.Timestamp],
                 initial_capital: float,
                 final_capital: float,
                 trades: List[Trade]):
        self.strategy = strategy
        self.portfolio_values = portfolio_values
        self.timestamps = timestamps
        self.initial_capital = initial_capital
        self.final_capital = final_capital
        self.trades = trades
        
        # Calculate derived metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.portfolio_values:
            self._set_default_metrics()
            return
        
        # Convert to numpy arrays
        values = np.array(self.portfolio_values)
        
        # Basic metrics
        self.total_return = (self.final_capital - self.initial_capital) / self.initial_capital
        self.total_pnl = self.final_capital - self.initial_capital
        
        # Returns
        returns = np.diff(values) / values[:-1]
        self.returns = returns
        
        # Volatility (annualized)
        self.volatility = np.std(returns) * np.sqrt(252 * 288 / 5)  # 5-minute data
        
        # Sharpe ratio (assuming risk-free rate = 0)
        self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 288 / 5) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        self.max_drawdown = self._calculate_max_drawdown(values)
        
        # Calmar ratio
        self.calmar_ratio = self.total_return / abs(self.max_drawdown) if self.max_drawdown != 0 else 0
        
        # Annualized return
        if len(self.timestamps) > 1:
            days = (self.timestamps[-1] - self.timestamps[0]).days
            self.annualized_return = (1 + self.total_return) ** (365 / days) - 1 if days > 0 else 0
        else:
            self.annualized_return = 0
        
        # Trade-specific metrics
        self.num_trades = len(self.trades)
        self.win_rate = self._calculate_win_rate()
        self.profit_factor = self._calculate_profit_factor()
        self.avg_trade_duration = self._calculate_avg_trade_duration()
        self.avg_trade_pnl = np.mean([trade.pnl for trade in self.trades]) if self.trades else 0
    
    def _set_default_metrics(self):
        """Set default metrics when no data is available."""
        self.total_return = 0
        self.total_pnl = 0
        self.returns = np.array([])
        self.volatility = 0
        self.sharpe_ratio = 0
        self.max_drawdown = 0
        self.calmar_ratio = 0
        self.annualized_return = 0
        self.num_trades = 0
        self.win_rate = 0
        self.profit_factor = 0
        self.avg_trade_duration = 0
        self.avg_trade_pnl = 0
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(values) == 0:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate of trades."""
        if not self.trades:
            return 0
        
        winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
        return winning_trades / len(self.trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        if not self.trades:
            return 0
        
        gross_profit = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in self.trades if trade.pnl < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in minutes."""
        if not self.trades:
            return 0
        
        durations = []
        for trade in self.trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
                durations.append(duration)
        
        return np.mean(durations) if durations else 0
    
    def plot_portfolio_value(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot portfolio value over time.
        
        Args:
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot portfolio value
        ax.plot(self.timestamps, self.portfolio_values, linewidth=2, color='blue')
        ax.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        
        # Add trade markers
        for trade in self.trades:
            if trade.entry_time in self.timestamps:
                idx = self.timestamps.index(trade.entry_time)
                ax.plot(trade.entry_time, self.portfolio_values[idx], 'g^', markersize=8, alpha=0.7)
            
            if trade.exit_time and trade.exit_time in self.timestamps:
                idx = self.timestamps.index(trade.exit_time)
                color = 'red' if trade.pnl < 0 else 'green'
                ax.plot(trade.exit_time, self.portfolio_values[idx], 'v', color=color, markersize=8, alpha=0.7)
        
        ax.set_title(f'Portfolio Value Over Time\nTotal Return: {self.total_return:.2%}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot drawdown over time.
        
        Args:
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate drawdown
        values = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        
        # Plot drawdown
        ax.fill_between(self.timestamps, 0, -drawdown, alpha=0.3, color='red')
        ax.plot(self.timestamps, -drawdown, color='red', linewidth=1)
        
        ax.set_title(f'Drawdown Over Time\nMax Drawdown: {self.max_drawdown:.2%}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trade_distribution(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot trade PnL distribution.
        
        Args:
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        if not self.trades:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PnL distribution
        pnls = [trade.pnl for trade in self.trades]
        
        ax1.hist(pnls, bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Trade PnL Distribution')
        ax1.set_xlabel('PnL')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative PnL
        cumulative_pnl = np.cumsum(pnls)
        ax2.plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl, linewidth=2)
        ax2.set_title('Cumulative PnL')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative PnL')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_monthly_returns(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot monthly returns heatmap.
        
        Args:
            save_path: Optional path to save plot
            
        Returns:
            Matplotlib figure
        """
        if len(self.timestamps) < 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Insufficient data for monthly returns', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create DataFrame with portfolio values
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'portfolio_value': self.portfolio_values
        })
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to monthly
        monthly = df.resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        # Create pivot table for heatmap
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        pivot_table = monthly_returns.pivot(index='year', columns='month', values='portfolio_value')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='RdYlGn', 
                   center=0, ax=ax, cbar_kws={'label': 'Monthly Return'})
        
        ax.set_title('Monthly Returns Heatmap')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'num_trades': self.num_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'avg_trade_duration': self.avg_trade_duration,
            'avg_trade_pnl': self.avg_trade_pnl,
            'total_pnl': self.total_pnl
        }
    
    def save_to_file(self, filepath: Path) -> None:
        """Save results to JSON file."""
        results_dict = {
            'summary_stats': self.get_summary_stats(),
            'trades': [{
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'direction': trade.direction,
                'pnl': trade.pnl,
                'fees': trade.fees
            } for trade in self.trades],
            'portfolio_timeline': [{
                'timestamp': ts.isoformat(),
                'value': val
            } for ts, val in zip(self.timestamps, self.portfolio_values)]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Saved backtest results to {filepath}")
    
    def load_from_file(self, filepath: Path) -> None:
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load summary stats
        stats = data['summary_stats']
        for key, value in stats.items():
            setattr(self, key, value)
        
        # Load trades
        self.trades = []
        for trade_data in data['trades']:
            trade = Trade(
                entry_time=pd.to_datetime(trade_data['entry_time']),
                exit_time=pd.to_datetime(trade_data['exit_time']) if trade_data['exit_time'] else None,
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                quantity=trade_data['quantity'],
                direction=trade_data['direction'],
                status='closed',
                pnl=trade_data['pnl'],
                fees=trade_data['fees']
            )
            self.trades.append(trade)
        
        # Load portfolio timeline
        self.timestamps = [pd.to_datetime(item['timestamp']) for item in data['portfolio_timeline']]
        self.portfolio_values = [item['value'] for item in data['portfolio_timeline']]
        
        logger.info(f"Loaded backtest results from {filepath}")
    
    def __str__(self) -> str:
        """String representation of results."""
        return f"""
Backtest Results Summary:
------------------------
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Volatility: {self.volatility:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}
Calmar Ratio: {self.calmar_ratio:.2f}

Trade Statistics:
-----------------
Number of Trades: {self.num_trades}
Win Rate: {self.win_rate:.2%}
Profit Factor: {self.profit_factor:.2f}
Average Trade Duration: {self.avg_trade_duration:.1f} minutes
Average Trade PnL: {self.avg_trade_pnl:.2f}
Total PnL: {self.total_pnl:.2f}
"""