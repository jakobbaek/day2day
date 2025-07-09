"""Main entry point for day2day application."""

from day2day.api.main import Day2DayAPI
from day2day.config.settings import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    print("Day2Day Trading Application")
    print("=" * 50)
    
    # Initialize API
    api = Day2DayAPI()
    
    # Show project status
    status = api.get_project_status()
    
    print(f"Raw Data Files: {len(status['raw_data_files'])}")
    print(f"Processed Data Files: {len(status['processed_data_files'])}")
    print(f"Model Suites: {len(status['model_suites'])}")
    print(f"Backtest Results: {len(status['backtest_results'])}")
    print()
    
    # Show available models
    available_models = api.get_available_models()
    print(f"Available Models: {', '.join(available_models)}")
    print()
    
    # Show configuration
    print("Configuration:")
    print(f"  Data Directory: {settings.data_dir}")
    print(f"  API Delay: {settings.api_delay}s")
    print(f"  Prediction Horizon: {settings.prediction_horizon_hours} hours")
    print(f"  Default Currency: {settings.default_currency}")
    print()
    
    print("Use the CLI interface (day2day.api.cli) for specific operations.")
    print("Example: python -m day2day.api.cli status")


if __name__ == '__main__':
    main()