"""Command line interface for day2day application."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any
from .main import Day2DayAPI
from ..config.settings import settings


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Day2Day Trading Application")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Authentication
    auth_parser = subparsers.add_parser('auth', help='Authenticate with Saxo Bank API')
    auth_parser.add_argument('--check', action='store_true', help='Check current authentication status')
    
    # Market data collection
    collect_parser = subparsers.add_parser('collect', help='Collect market data')
    collect_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    collect_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    collect_parser.add_argument('--output-file', default='danish_stocks_1m.csv', help='Output file name')
    collect_parser.add_argument('--update', action='store_true', help='Update existing data')
    
    # Data preparation
    prepare_parser = subparsers.add_parser('prepare', help='Prepare training data')
    prepare_parser.add_argument('--raw-data-file', required=True, help='Raw data file name')
    prepare_parser.add_argument('--output-title', required=True, help='Output title')
    prepare_parser.add_argument('--target-instrument', required=True, help='Target instrument')
    prepare_parser.add_argument('--target-price-type', default='high', help='Price type to predict (NOTE: Always forced to "high")')
    prepare_parser.add_argument('--use-percentage', action='store_true', help='Use percentage features instead of raw prices (mutually exclusive)')
    prepare_parser.add_argument('--no-standardize-datetime', action='store_true', help='Disable datetime standardization')
    prepare_parser.add_argument('--volume-fraction', type=float, default=1.0, help='Fraction of most traded stocks to include (0.0-1.0, e.g., 0.25 for top 25%)')
    
    # Model training
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--training-data-title', required=True, help='Training data title')
    train_parser.add_argument('--target-instrument', required=True, help='Target instrument')
    train_parser.add_argument('--models', nargs='+', help='Models to train')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    # Bootstrapping
    bootstrap_parser = subparsers.add_parser('bootstrap', help='Run bootstrap analysis')
    bootstrap_parser.add_argument('--training-data-title', required=True, help='Training data title')
    bootstrap_parser.add_argument('--target-instrument', required=True, help='Target instrument')
    bootstrap_parser.add_argument('--model-type', required=True, help='Model type')
    bootstrap_parser.add_argument('--n-bootstrap', type=int, default=100, help='Number of bootstrap samples')
    
    # Model evaluation
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    evaluate_parser.add_argument('--training-data-title', required=True, help='Training data title')
    evaluate_parser.add_argument('--target-instrument', required=True, help='Target instrument')
    evaluate_parser.add_argument('--create-plots', action='store_true', help='Create visualization plots')
    
    # Backtesting
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest')
    backtest_parser.add_argument('--training-data-title', required=True, help='Training data title')
    backtest_parser.add_argument('--target-instrument', required=True, help='Target instrument')
    backtest_parser.add_argument('--model-name', required=True, help='Model name')
    backtest_parser.add_argument('--strategy-params', type=str, help='Strategy parameters as JSON')
    backtest_parser.add_argument('--strategy-name', default='default', help='Strategy name')
    
    # Status
    status_parser = subparsers.add_parser('status', help='Get project status')
    
    # Cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--days-old', type=int, default=30, help='Days old threshold')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize API
    api = Day2DayAPI()
    
    try:
        if args.command == 'auth':
            if args.check:
                # Check authentication status
                if api.check_authentication():
                    print("✓ Authentication is valid")
                else:
                    print("✗ Authentication is invalid or expired")
                    print("Run 'day2day auth' to authenticate")
            else:
                # Perform authentication
                print("Starting Saxo Bank API authentication...")
                if api.authenticate_saxo_bank():
                    print("✓ Authentication completed successfully")
                else:
                    print("✗ Authentication failed")
                    sys.exit(1)
        
        elif args.command == 'collect':
            api.collect_market_data(
                start_date=args.start_date,
                end_date=args.end_date,
                output_file=args.output_file,
                update_existing=args.update
            )
            print(f"Market data collected successfully to {args.output_file}")
        
        elif args.command == 'prepare':
            output_path = api.prepare_training_data(
                raw_data_file=args.raw_data_file,
                output_title=args.output_title,
                target_instrument=args.target_instrument,
                target_price_type=args.target_price_type,
                use_percentage_features=args.use_percentage,
                standardize_datetime=not args.no_standardize_datetime,
                volume_fraction=args.volume_fraction
            )
            print(f"Training data prepared successfully: {output_path}")
            print("✓ Target prediction: HIGH price of target instrument (as per specifications)")
            if not args.no_standardize_datetime:
                print("✓ Datetime standardization: GMT conversion and complete timeline creation enabled")
            if args.volume_fraction < 1.0:
                print(f"✓ Trading volume filter: Top {args.volume_fraction*100:.1f}% most traded stocks selected")
            if args.use_percentage:
                print("✓ Price features: Using percentage changes (same column names)")
            else:
                print("✓ Price features: Using raw price values (default)")
        
        elif args.command == 'train':
            # Create model configs
            model_configs = None
            if args.models:
                model_configs = {}
                for model in args.models:
                    model_configs[model] = {'type': model, 'params': {}}
            
            models = api.train_models(
                training_data_title=args.training_data_title,
                target_instrument=args.target_instrument,
                model_configs=model_configs,
                test_size=args.test_size
            )
            print(f"Trained {len(models)} models successfully")
        
        elif args.command == 'bootstrap':
            results = api.run_bootstrap_analysis(
                training_data_title=args.training_data_title,
                target_instrument=args.target_instrument,
                model_type=args.model_type,
                model_params={},
                n_bootstrap=args.n_bootstrap
            )
            print(f"Bootstrap analysis completed with {results['n_bootstrap']} samples")
        
        elif args.command == 'evaluate':
            results = api.evaluate_models(
                training_data_title=args.training_data_title,
                target_instrument=args.target_instrument
            )
            
            print("Model Evaluation Results:")
            print("=" * 50)
            for model_name, model_results in results.items():
                metrics = model_results['metrics']
                print(f"{model_name}:")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  MAPE: {metrics['mape']:.2f}%")
                print()
            
            if args.create_plots:
                api.create_prediction_plots(
                    args.training_data_title,
                    args.target_instrument
                )
                print("Visualization plots created successfully")
        
        elif args.command == 'backtest':
            # Parse strategy parameters
            strategy_params = {}
            if args.strategy_params:
                strategy_params = json.loads(args.strategy_params)
            
            # Default strategy parameters
            default_params = {
                'initial_capital': 100000,
                'min_probability': 0.6,
                'price_increase_threshold': 0.02,
                'take_profit_threshold': 0.03,
                'stop_loss_threshold': 0.02,
                'max_positions': 4
            }
            
            # Merge with defaults
            final_params = {**default_params, **strategy_params}
            
            results = api.run_backtest(
                training_data_title=args.training_data_title,
                target_instrument=args.target_instrument,
                model_name=args.model_name,
                strategy_params=final_params,
                strategy_name=args.strategy_name
            )
            
            print("Backtest Results:")
            print("=" * 30)
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"Number of Trades: {results['num_trades']}")
            print(f"Win Rate: {results['win_rate']:.2%}")
            print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        elif args.command == 'status':
            status = api.get_project_status()
            
            print("Project Status:")
            print("=" * 30)
            print(f"Raw Data Files: {len(status['raw_data_files'])}")
            for file in status['raw_data_files']:
                print(f"  - {file}")
            
            print(f"Processed Data Files: {len(status['processed_data_files'])}")
            for file in status['processed_data_files']:
                print(f"  - {file}")
            
            print(f"Model Suites: {len(status['model_suites'])}")
            for suite in status['model_suites']:
                print(f"  - {suite}")
            
            print(f"Backtest Results: {len(status['backtest_results'])}")
            for result in status['backtest_results']:
                print(f"  - {result}")
        
        elif args.command == 'cleanup':
            api.cleanup_old_data(args.days_old)
            print(f"Cleaned up data older than {args.days_old} days")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()