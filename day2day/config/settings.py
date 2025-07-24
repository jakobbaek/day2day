"""Configuration settings for day2day application."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import json

# Load environment variables from .env file
# Try to load from project root first, then current directory
try:
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # Try current directory or system env
except Exception:
    # If loading fails, continue without .env (use system env vars)
    pass

class Settings:
    """Application settings management."""
    
    def __init__(self):
        # Use MAIN_PATH from environment if set, otherwise use project root
        main_path = os.getenv("MAIN_PATH")
        if main_path:
            self.base_dir = Path(main_path).expanduser().resolve()
        else:
            self.base_dir = Path(__file__).parent.parent.parent
        
        self.data_dir = self.base_dir / "data"
        self.config_dir = self.base_dir / "config"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "models").mkdir(exist_ok=True)
        (self.data_dir / "backtests").mkdir(exist_ok=True)
        
    # Saxo Bank API Configuration
    @property
    def saxo_client_id(self) -> str:
        """Get Saxo Bank client ID from environment."""
        return os.getenv("SAXO_CLIENT_ID", "")
    
    @property
    def saxo_client_secret(self) -> str:
        """Get Saxo Bank client secret from environment."""
        return os.getenv("SAXO_CLIENT_SECRET", "")
    
    @property
    def saxo_redirect_uri(self) -> str:
        """Get Saxo Bank redirect URI from environment."""
        return os.getenv("SAXO_REDIRECT_URI", "https://jakobogtherese.dk")
    
    @property
    def saxo_access_token(self) -> str:
        """Get Saxo Bank access token from environment."""
        return os.getenv("SAXO_ACCESS_TOKEN", "")
    
    @property
    def saxo_refresh_token(self) -> str:
        """Get Saxo Bank refresh token from environment."""
        return os.getenv("SAXO_REFRESH_TOKEN", "")
    
    @property
    def saxo_base_url(self) -> str:
        """Get Saxo Bank API base URL."""
        return os.getenv("SAXO_BASE_URL", "https://gateway.saxobank.com/sim/openapi")
    
    @property
    def saxo_auth_url(self) -> str:
        """Get Saxo Bank authorization URL."""
        return os.getenv("SAXO_AUTH_URL", "https://sim.logonvalidation.net")
    
    # Data Collection Settings
    @property
    def api_delay(self) -> float:
        """API request delay to avoid rate limits."""
        return float(os.getenv("API_DELAY", "0.95"))
    
    @property
    def data_interval_days(self) -> int:
        """Data collection interval in days."""
        return int(os.getenv("DATA_INTERVAL_DAYS", "3"))
    
    @property
    def default_exchange(self) -> str:
        """Default exchange for data collection."""
        return os.getenv("DEFAULT_EXCHANGE", "CSE")  # Copenhagen Stock Exchange
    
    # Model Training Settings
    @property
    def prediction_horizon_hours(self) -> int:
        """Prediction horizon in hours."""
        return int(os.getenv("PREDICTION_HORIZON_HOURS", "2"))
    
    @property
    def prediction_horizon_minutes(self) -> int:
        """Prediction horizon in minutes (for 1-minute intervals)."""
        return self.prediction_horizon_hours * 60  # 2 hours = 120 1-minute intervals
    
    @property
    def default_train_test_split(self) -> float:
        """Default train/test split ratio."""
        return float(os.getenv("DEFAULT_TRAIN_TEST_SPLIT", "0.8"))
    
    # Backtesting Settings
    @property
    def default_currency(self) -> str:
        """Default currency for backtesting."""
        return os.getenv("DEFAULT_CURRENCY", "DKK")
    
    @property
    def default_exchange_fee(self) -> float:
        """Default exchange fee as fraction."""
        return float(os.getenv("DEFAULT_EXCHANGE_FEE", "0.0008"))  # 0.08%
    
    @property
    def default_trade_cost(self) -> float:
        """Default fixed trade cost."""
        return float(os.getenv("DEFAULT_TRADE_COST", "20.0"))
    
    # File Paths
    @property
    def raw_data_path(self) -> Path:
        """Path to raw data directory."""
        return self.data_dir / "raw"
    
    @property
    def processed_data_path(self) -> Path:
        """Path to processed data directory."""
        return self.data_dir / "processed"
    
    @property
    def models_path(self) -> Path:
        """Path to models directory."""
        return self.data_dir / "models"
    
    @property
    def backtests_path(self) -> Path:
        """Path to backtests directory."""
        return self.data_dir / "backtests"
    
    def get_raw_data_file(self, filename: str) -> Path:
        """Get full path for raw data file."""
        return self.raw_data_path / filename
    
    def get_processed_data_file(self, filename: str) -> Path:
        """Get full path for processed data file."""
        return self.processed_data_path / filename
    
    def get_model_file(self, filename: str) -> Path:
        """Get full path for model file."""
        return self.models_path / filename
    
    def get_backtest_file(self, filename: str) -> Path:
        """Get full path for backtest file."""
        return self.backtests_path / filename
    
    def save_config(self, config: Dict[str, Any], filename: str) -> None:
        """Save configuration to JSON file."""
        config_path = self.config_dir / filename
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load configuration from JSON file."""
        config_path = self.config_dir / filename
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return None

# Global settings instance
settings = Settings()