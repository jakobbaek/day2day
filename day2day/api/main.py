"""Main API interface for day2day application."""

from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from ..data.market_data import MarketDataCollector
from ..data.preparation import DataPreparator
from ..models.training import ModelTrainer
from ..models.bootstrap import BootstrapEstimator
from ..models.evaluation import ModelEvaluator
from ..backtesting.backtester import Backtester
from ..backtesting.strategy import ProbabilityBasedStrategy, ModelBasedStrategy
from ..config.settings import settings

logger = logging.getLogger(__name__)


class Day2DayAPI:
    """Main API class for day2day trading application."""
    
    def __init__(self):
        self.market_collector = MarketDataCollector()
        self.data_preparator = DataPreparator()
        self.model_trainer = ModelTrainer()
        self.bootstrap_estimator = BootstrapEstimator()
        self.model_evaluator = ModelEvaluator()
        self.backtester = Backtester()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
    # Authentication
    def authenticate_saxo_bank(self) -> bool:
        """
        Authenticate with Saxo Bank API using interactive OAuth flow.
        
        Returns:
            True if authentication successful, False otherwise
        """
        try:
            self.market_collector.authenticator.get_access_token_interactive()
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False
    
    def check_authentication(self) -> bool:
        """
        Check if current authentication is valid.
        
        Returns:
            True if authenticated, False otherwise
        """
        return self.market_collector.authenticator.is_token_valid()
    
    def ensure_authentication(self) -> bool:
        """
        Ensure authentication is valid, prompting user if needed.
        
        Returns:
            True if authentication is valid, False otherwise
        """
        return self.market_collector.authenticator.ensure_valid_token()
    
    # Market Data Collection
    def collect_market_data(self, 
                          start_date: str, 
                          end_date: str,
                          output_file: str = "danish_stocks_1m.csv",
                          update_existing: bool = False) -> None:
        """
        Collect market data from Saxo Bank API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_file: Output CSV filename
            update_existing: Whether to update existing data
        """
        logger.info(f"Collecting market data from {start_date} to {end_date}")
        
        self.market_collector.collect_market_data(
            start_date=start_date,
            end_date=end_date,
            output_file=output_file,
            update_existing=update_existing
        )
        
        logger.info("Market data collection completed")
    
    def get_available_instruments(self) -> List[Dict[str, Any]]:
        """
        Get list of available instruments.
        
        Returns:
            List of instrument dictionaries
        """
        df = self.market_collector.get_instrument_list()
        return df.to_dicts()
    
    # Data Preparation
    def prepare_training_data(self, 
                            raw_data_file: str,
                            output_title: str,
                            target_instrument: str,
                            target_price_type: str = "high",
                            standardize_datetime: bool = True,
                            exclude_last_hours: float = 0.0,
                            **kwargs) -> str:
        """
        Prepare training data for model training with enhanced datetime standardization.
        
        Note: As per specifications, the target is always the HIGH price of the target instrument.
        
        Args:
            raw_data_file: Name of raw data file
            output_title: Title for output file
            target_instrument: Target instrument for prediction
            target_price_type: Type of price to predict (NOTE: Always forced to "high")
            standardize_datetime: Whether to standardize datetime to GMT and create complete timeline
            exclude_last_hours: Hours to exclude from end of each trading day to avoid next-day predictions
            **kwargs: Additional preparation parameters
            
        Returns:
            Path to prepared data file
        """
        logger.info(f"Preparing training data: {output_title} (standardize_datetime={standardize_datetime})")
        
        return self.data_preparator.prepare_training_data(
            raw_data_file=raw_data_file,
            output_title=output_title,
            target_instrument=target_instrument,
            target_price_type=target_price_type,
            exclude_last_hours=exclude_last_hours,
            **kwargs
        )
    
    def update_training_data(self, title: str, new_raw_data_file: str) -> str:
        """
        Update existing training data with new raw data.
        
        Args:
            title: Title of existing training data
            new_raw_data_file: New raw data file
            
        Returns:
            Path to updated training data file
        """
        return self.data_preparator.update_training_data(title, new_raw_data_file)
    
    # Model Training
    def train_models(self, 
                    training_data_title: str,
                    target_instrument: str,
                    model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Train a suite of models.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            model_configs: Model configurations (None for defaults)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary of trained models
        """
        logger.info(f"Training models for {training_data_title}_{target_instrument}")
        logger.info(f"Models will predict HIGH price of {target_instrument} (as per specifications)")
        
        # Use recommended models if no configs provided
        if model_configs is None:
            # Load data to get recommendations (preserve nulls for day trading)
            data_file = f"{training_data_title}.csv"
            X, y = self.model_trainer.load_training_data(data_file, preserve_nulls=True)
            
            recommendations = self.model_trainer.get_model_recommendations(
                len(X.columns), len(X), day_trading_mode=True
            )
            
            model_configs = {rec['name']: {'type': rec['type'], 'params': rec['params']} 
                           for rec in recommendations}
        
        return self.model_trainer.train_model_suite(
            training_data_title=training_data_title,
            target_instrument=target_instrument,
            model_configs=model_configs,
            **kwargs
        )
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return self.model_trainer.list_available_models()
    
    # Bootstrapping
    def run_bootstrap_analysis(self, 
                             training_data_title: str,
                             target_instrument: str,
                             model_type: str,
                             model_params: Dict[str, Any],
                             n_bootstrap: int = 100) -> Dict[str, Any]:
        """
        Run bootstrap analysis for uncertainty estimation.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            model_type: Type of model to bootstrap
            model_params: Model parameters
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap analysis results
        """
        logger.info(f"Running bootstrap analysis with {n_bootstrap} samples")
        
        self.bootstrap_estimator.n_bootstrap = n_bootstrap
        
        return self.bootstrap_estimator.run_bootstrap_analysis(
            training_data_title=training_data_title,
            target_instrument=target_instrument,
            model_type=model_type,
            model_params=model_params
        )
    
    def get_prediction_probability(self, 
                                 training_data_title: str,
                                 target_instrument: str,
                                 model_type: str,
                                 threshold: float) -> List[float]:
        """
        Get probability that predictions exceed threshold.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            model_type: Model type
            threshold: Threshold value
            
        Returns:
            Array of probabilities
        """
        probabilities = self.bootstrap_estimator.get_probability_above_threshold(
            training_data_title, target_instrument, model_type, threshold
        )
        return probabilities.tolist()
    
    # Model Evaluation
    def evaluate_models(self, 
                       training_data_title: str,
                       target_instrument: str) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models in a suite.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            
        Returns:
            Evaluation results for each model
        """
        logger.info(f"Evaluating models for {training_data_title}_{target_instrument}")
        
        return self.model_evaluator.evaluate_model_suite(
            training_data_title, target_instrument
        )
    
    def create_prediction_plots(self, 
                              training_data_title: str,
                              target_instrument: str,
                              model_names: Optional[List[str]] = None) -> None:
        """
        Create prediction visualization plots.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            model_names: List of model names (None for all)
        """
        logger.info("Creating prediction plots")
        
        self.model_evaluator.create_prediction_plot(
            training_data_title, target_instrument, model_names
        )
        
        self.model_evaluator.create_scatter_plot(
            training_data_title, target_instrument, model_names
        )
        
        # DIAGNOSTIC: Check for data bleeding across market boundaries
        self.model_evaluator.create_daily_timeline_diagnostic(
            training_data_title, target_instrument, model_names
        )
        
        self.model_evaluator.create_metrics_comparison(
            training_data_title, target_instrument
        )
    
    def get_best_model(self, 
                      training_data_title: str,
                      target_instrument: str,
                      metric: str = 'rmse') -> tuple:
        """
        Get the best model based on a metric.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            metric: Metric to optimize
            
        Returns:
            Tuple of (model_name, metrics)
        """
        return self.model_evaluator.get_best_model(
            training_data_title, target_instrument, metric
        )
    
    # Backtesting
    def run_backtest(self, 
                    training_data_title: str,
                    target_instrument: str,
                    model_name: str,
                    strategy_params: Dict[str, Any],
                    strategy_name: str = "default") -> Dict[str, Any]:
        """
        Run backtest with model-based strategy.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            model_name: Name of model to use
            strategy_params: Strategy parameters
            strategy_name: Name for strategy
            
        Returns:
            Backtest results summary
        """
        logger.info(f"Running backtest for {model_name} strategy")
        
        # Load test data
        test_data = self.backtester.load_test_data(training_data_title, target_instrument)
        
        # Create strategy
        strategy = ModelBasedStrategy(
            training_data_title=training_data_title,
            target_instrument=target_instrument,
            model_name=model_name,
            **strategy_params
        )
        
        # Run backtest
        results = self.backtester.run_backtest(strategy, test_data)
        
        # Save results
        self.backtester.save_backtest_results(
            results, training_data_title, target_instrument, strategy_name
        )
        
        return results.get_summary_stats()
    
    def run_parameter_sweep(self, 
                          training_data_title: str,
                          target_instrument: str,
                          model_name: str,
                          base_params: Dict[str, Any],
                          param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Run parameter sweep for strategy optimization.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            model_name: Name of model to use
            base_params: Base strategy parameters
            param_ranges: Parameter ranges to test
            
        Returns:
            List of results for each parameter combination
        """
        logger.info("Running parameter sweep")
        
        # Load test data
        test_data = self.backtester.load_test_data(training_data_title, target_instrument)
        
        # Create strategy class with model info
        def create_strategy(**params):
            return ModelBasedStrategy(
                training_data_title=training_data_title,
                target_instrument=target_instrument,
                model_name=model_name,
                **params
            )
        
        return self.backtester.run_parameter_sweep(
            create_strategy, base_params, param_ranges, test_data
        )
    
    def compare_strategies(self, 
                         training_data_title: str,
                         target_instrument: str,
                         strategy_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Args:
            training_data_title: Title of training data
            target_instrument: Target instrument
            strategy_configs: List of strategy configurations
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(strategy_configs)} strategies")
        
        # Load test data
        test_data = self.backtester.load_test_data(training_data_title, target_instrument)
        
        # Create strategies
        strategies = []
        strategy_names = []
        
        for config in strategy_configs:
            strategy = ModelBasedStrategy(
                training_data_title=training_data_title,
                target_instrument=target_instrument,
                **config
            )
            strategies.append(strategy)
            strategy_names.append(config.get('name', 'unnamed'))
        
        # Compare strategies
        comparison_df = self.backtester.compare_strategies(
            strategies, strategy_names, test_data
        )
        
        return comparison_df.to_dict('records')
    
    # Utility Methods
    def get_project_status(self) -> Dict[str, Any]:
        """Get overall project status."""
        status = {
            'raw_data_files': [],
            'processed_data_files': [],
            'model_suites': [],
            'backtest_results': []
        }
        
        # Check raw data
        raw_data_dir = settings.raw_data_path
        if raw_data_dir.exists():
            status['raw_data_files'] = [f.name for f in raw_data_dir.glob("*.csv")]
        
        # Check processed data
        processed_data_dir = settings.processed_data_path
        if processed_data_dir.exists():
            status['processed_data_files'] = [f.name for f in processed_data_dir.glob("*.csv")]
        
        # Check model suites
        models_dir = settings.models_path
        if models_dir.exists():
            status['model_suites'] = [d.name for d in models_dir.iterdir() if d.is_dir()]
        
        # Check backtest results
        backtests_dir = settings.backtests_path
        if backtests_dir.exists():
            status['backtest_results'] = [d.name for d in backtests_dir.iterdir() if d.is_dir()]
        
        return status
    
    def cleanup_old_data(self, days_old: int = 30) -> None:
        """
        Clean up old data files.
        
        Args:
            days_old: Delete files older than this many days
        """
        import time
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cutoff_timestamp = cutoff_date.timestamp()
        
        # Clean up raw data
        for file_path in settings.raw_data_path.glob("*.csv"):
            if file_path.stat().st_mtime < cutoff_timestamp:
                file_path.unlink()
                logger.info(f"Deleted old raw data file: {file_path}")
        
        # Clean up processed data
        for file_path in settings.processed_data_path.glob("*.csv"):
            if file_path.stat().st_mtime < cutoff_timestamp:
                file_path.unlink()
                logger.info(f"Deleted old processed data file: {file_path}")
        
        logger.info(f"Cleanup completed for files older than {days_old} days")