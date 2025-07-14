"""Model training and management module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from pathlib import Path
from .base import BaseModel, ModelEnsemble
from .implementations import create_model, get_available_models, get_null_compatible_models
from ..config.settings import settings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and management."""
    
    def __init__(self):
        self.settings = settings
        self.trained_models: Dict[str, BaseModel] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        
    def load_training_data(self, data_file: str, preserve_nulls: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load training data from processed data file.
        
        Args:
            data_file: Name of the processed data file
            preserve_nulls: Whether to preserve null values for day trading models
            
        Returns:
            Tuple of (features, target)
        """
        file_path = settings.get_processed_data_file(data_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Separate features and target
        if 'target' not in df.columns:
            raise ValueError("Target column not found in training data")
        
        # Exclude non-feature columns
        exclude_cols = ['datetime', 'target', 'trading_eligible']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['target']
        
        if preserve_nulls:
            # Keep nulls for day trading models (only remove rows where target is null)
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
            
            null_count = X.isnull().sum().sum()
            logger.info(f"Loaded training data (preserving nulls): {len(X)} samples, {len(feature_cols)} features")
            logger.info(f"Null values preserved: {null_count} nulls in feature matrix (important for day trading)")
        else:
            # Remove rows with missing values (for traditional models)
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            logger.info(f"Loaded training data (nulls removed): {len(X)} samples, {len(feature_cols)} features")
        
        return X, y
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                               test_size: float = 0.2, 
                               method: str = 'temporal') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create train/test split.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of data for testing
            method: Split method ('temporal' or 'random')
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if method == 'temporal':
            # Split based on time order (last test_size for testing)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            # Random split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, model_type: str, model_name: str, 
                   X_train: pd.DataFrame, y_train: pd.Series,
                   **model_params) -> BaseModel:
        """
        Train a single model.
        
        Args:
            model_type: Type of model to train
            model_name: Name for the model
            X_train: Training features
            y_train: Training target
            **model_params: Model-specific parameters
            
        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model: {model_name}")
        
        # Create model
        model = create_model(model_type, name=model_name, **model_params)
        
        # Train model
        model.train(X_train, y_train)
        
        # Store model and config
        self.trained_models[model_name] = model
        self.model_configs[model_name] = {
            'type': model_type,
            'params': model_params,
            'trained': True
        }
        
        return model
    
    def train_model_suite(self, 
                         training_data_title: str,
                         target_instrument: str,
                         model_configs: Dict[str, Dict[str, Any]],
                         test_size: float = 0.2,
                         split_method: str = 'temporal') -> Dict[str, BaseModel]:
        """
        Train a suite of models.
        
        Note: As per specifications, models always predict the HIGH price of the target instrument.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument being predicted (HIGH price)
            model_configs: Dictionary of model configurations
            test_size: Test set size
            split_method: Method for train/test split
            
        Returns:
            Dictionary of trained models
        """
        logger.info(f"Training model suite for {training_data_title}")
        logger.info(f"Target prediction: HIGH price of {target_instrument} (as per specifications)")
        
        # Load training data
        data_file = f"{training_data_title}.csv"
        X, y = self.load_training_data(data_file)
        
        # Create train/test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(
            X, y, test_size=test_size, method=split_method
        )
        
        # Train each model
        trained_models = {}
        
        for model_name, config in model_configs.items():
            model_type = config['type']
            model_params = config.get('params', {})
            
            try:
                model = self.train_model(
                    model_type, model_name, X_train, y_train, **model_params
                )
                trained_models[model_name] = model
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Save models and metadata
        self.save_model_suite(training_data_title, target_instrument, trained_models, X_test, y_test)
        
        return trained_models
    
    def save_model_suite(self, training_data_title: str, target_instrument: str,
                        models: Dict[str, BaseModel], X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Save trained models and metadata.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
        """
        suite_name = f"{training_data_title}_{target_instrument}"
        suite_dir = settings.models_path / suite_name
        suite_dir.mkdir(exist_ok=True)
        
        # Save individual models
        for model_name, model in models.items():
            model_file = suite_dir / f"{model_name}.joblib"
            model.save(model_file)
        
        # Save test data
        test_file = suite_dir / "test_data.csv"
        test_df = X_test.copy()
        test_df['target'] = y_test
        
        # Include trading_eligible flag if it exists in the original data
        data_file = f"{training_data_title}.csv"
        original_df = pd.read_csv(settings.get_processed_data_file(data_file))
        if 'trading_eligible' in original_df.columns:
            # Get the trading_eligible values for the test set indices
            test_df['trading_eligible'] = original_df.loc[X_test.index, 'trading_eligible']
            logger.info("Included trading_eligible flag in test data for day trading evaluation")
        
        test_df.to_csv(test_file, index=False)
        
        # Save metadata
        metadata = {
            'training_data_title': training_data_title,
            'target_instrument': target_instrument,
            'models': list(models.keys()),
            'test_size': len(X_test),
            'train_size': len(X_test) * 4,  # Assuming 80/20 split
            'feature_count': len(X_test.columns),
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = suite_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved model suite to {suite_dir}")
    
    def load_model_suite(self, training_data_title: str, target_instrument: str) -> Dict[str, BaseModel]:
        """
        Load a saved model suite.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            
        Returns:
            Dictionary of loaded models
        """
        suite_name = f"{training_data_title}_{target_instrument}"
        suite_dir = settings.models_path / suite_name
        
        if not suite_dir.exists():
            raise FileNotFoundError(f"Model suite not found: {suite_dir}")
        
        # Load metadata
        metadata_file = suite_dir / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load models
        models = {}
        for model_name in metadata['models']:
            model_file = suite_dir / f"{model_name}.joblib"
            if model_file.exists():
                # Create empty model and load
                model = create_model('xgboost', name=model_name)  # Placeholder
                model.load(model_file)
                models[model_name] = model
        
        logger.info(f"Loaded {len(models)} models from {suite_dir}")
        
        return models
    
    def get_model_recommendations(self, feature_count: int, sample_count: int, 
                                 day_trading_mode: bool = True) -> List[Dict[str, Any]]:
        """
        Get recommended model configurations based on data characteristics.
        
        For day trading applications, only recommends models that can handle null values
        since lagged features will have nulls at market open.
        
        Args:
            feature_count: Number of features
            sample_count: Number of samples
            day_trading_mode: Whether to use only null-compatible models for day trading
            
        Returns:
            List of recommended model configurations
        """
        recommendations = []
        
        if day_trading_mode:
            logger.info("Day trading mode: Using single XGBoost model (default)")
            logger.info("XGBoost is optimal for day trading as it handles nulls at market open")
            
            # Get available null-compatible models
            null_safe_models = get_null_compatible_models()
            
            if 'xgboost' in null_safe_models:
                # Single XGBoost model as default
                recommendations.append({
                    'name': 'xgboost_default',
                    'type': 'xgboost',
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'random_state': 42
                    }
                })
            elif 'random_forest' in null_safe_models:
                # Fallback to Random Forest if XGBoost not available
                logger.warning("XGBoost not available, falling back to Random Forest")
                recommendations.append({
                    'name': 'random_forest_default',
                    'type': 'random_forest',
                    'params': {
                        'n_estimators': 100,
                        'max_depth': 10,
                        'random_state': 42
                    }
                })
            else:
                logger.error("No null-compatible models available! Install XGBoost or scikit-learn")
                
        else:
            # Traditional mode - single XGBoost model
            logger.info("Traditional mode: Using single XGBoost model (default)")
            
            # Single XGBoost model as default
            recommendations.append({
                'name': 'xgboost_default',
                'type': 'xgboost',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            })
        
        return recommendations
    
    def create_ensemble(self, models: Dict[str, BaseModel], 
                       weights: Optional[Dict[str, float]] = None) -> ModelEnsemble:
        """
        Create an ensemble from trained models.
        
        Args:
            models: Dictionary of trained models
            weights: Optional weights for ensemble
            
        Returns:
            Model ensemble
        """
        return ModelEnsemble(models, weights)
    
    def list_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(get_available_models().keys())