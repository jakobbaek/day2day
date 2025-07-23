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
            total_cells = len(X) * len(X.columns)
            null_percentage = (null_count / total_cells) * 100
            
            logger.info(f"Loaded training data (preserving nulls): {len(X)} samples, {len(feature_cols)} features")
            logger.info(f"Null values preserved: {null_count:,} / {total_cells:,} cells ({null_percentage:.2f}%)")
            
            # Break down nulls by type for better understanding
            if null_percentage > 50:
                logger.warning(f"High null percentage ({null_percentage:.1f}%) detected - analyzing...")
                
                # Check nulls per feature
                null_counts_per_feature = X.isnull().sum().sort_values(ascending=False)
                top_null_features = null_counts_per_feature.head(10)
                
                logger.info("Features with most nulls:")
                for feature, count in top_null_features.items():
                    pct = (count / len(X)) * 100
                    logger.info(f"  {feature}: {count:,} ({pct:.1f}%)")
                
                # Identify potential causes
                lag_features = [f for f in X.columns if 'lag' in f]
                historical_features = [f for f in X.columns if 'mean_' in f]
                
                if lag_features:
                    lag_nulls = X[lag_features].isnull().sum().sum()
                    lag_pct = (lag_nulls / (len(X) * len(lag_features))) * 100
                    logger.info(f"Lagged features: {lag_nulls:,} nulls ({lag_pct:.1f}%) - expected for day trading")
                
                if historical_features:
                    hist_nulls = X[historical_features].isnull().sum().sum()
                    hist_pct = (hist_nulls / (len(X) * len(historical_features))) * 100
                    logger.info(f"Historical features: {hist_nulls:,} nulls ({hist_pct:.1f}%) - may indicate holiday gaps")
            else:
                logger.info(f"Null percentage ({null_percentage:.1f}%) is reasonable for day trading")
            
            # CRITICAL DEBUG: Check target variable distribution
            target_std = y.std()
            target_var = y.var()
            target_range = y.max() - y.min()
            logger.info(f"Target variable stats: std={target_std:.6f}, var={target_var:.6f}, range={target_range:.6f}")
            
            if target_std < 1e-4:
                logger.error("CRITICAL: Target variable has very low variance - this WILL cause flat predictions!")
                logger.error(f"Target sample values: {y.head(10).tolist()}")
                logger.error("This indicates a data pipeline issue, not a model issue")
            
            # Debug: Check feature variance
            feature_vars = X.var(numeric_only=True)
            low_var_features = feature_vars[feature_vars < 1e-6]
            if len(low_var_features) > 0:
                logger.warning(f"Found {len(low_var_features)} features with very low variance")
                logger.warning(f"Low variance features: {low_var_features.head()}")
                
            # Debug: Check for constant features and feature quality
            constant_features = []
            low_unique_features = []
            
            for col in X.columns:
                unique_count = X[col].nunique()
                if unique_count <= 1:
                    constant_features.append(col)
                elif unique_count <= 5:  # Very few unique values
                    low_unique_features.append((col, unique_count))
            
            if constant_features:
                logger.error(f"CRITICAL: Found {len(constant_features)} constant features: {constant_features[:5]}")
                logger.error("Constant features provide no information - this could cause flat predictions!")
                
            if low_unique_features:
                logger.warning(f"Found {len(low_unique_features)} features with very few unique values:")
                for col, count in low_unique_features[:5]:
                    logger.warning(f"  {col}: {count} unique values")
            
            # Check if most features are just different versions of the same data
            numeric_features = X.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 10:
                # Sample correlations to see if features are too similar
                sample_features = numeric_features[:min(20, len(numeric_features))]
                corr_matrix = X[sample_features].corr()
                
                # Count highly correlated pairs
                high_corr_count = 0
                for i in range(len(corr_matrix)):
                    for j in range(i+1, len(corr_matrix)):
                        if abs(corr_matrix.iloc[i, j]) > 0.95:
                            high_corr_count += 1
                
                if high_corr_count > len(sample_features) // 2:
                    logger.warning(f"Many features are highly correlated ({high_corr_count} pairs > 0.95)")
                    logger.warning("This suggests features might be duplicates or very similar")
                    
            # Check if non-null features actually have meaningful variation
            meaningful_features = 0
            for col in numeric_features[:20]:  # Sample first 20 numeric features
                non_null_values = X[col].dropna()
                if len(non_null_values) > 100 and non_null_values.std() > 1e-6:
                    meaningful_features += 1
                    
            if meaningful_features < 5:
                logger.error(f"CRITICAL: Only {meaningful_features} features have meaningful variation!")
                logger.error("Most features may be constant or near-constant - this WILL cause flat predictions!")
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
            
            # CRITICAL DEBUG: Check if temporal split is causing flat predictions
            train_var = y_train.var()
            test_var = y_test.var()
            logger.info(f"Train/test target variance: train={train_var:.6f}, test={test_var:.6f}")
            
            if test_var < 1e-4:
                logger.error("CRITICAL: Test set has very low target variance!")
                logger.error("This means temporal split put all similar data in test set")
                logger.error(f"Test target range: {y_test.min():.6f} to {y_test.max():.6f}")
                logger.error(f"Test target sample: {y_test.head(10).tolist()}")
                
            if train_var < 1e-4:
                logger.error("CRITICAL: Train set has very low target variance!")
                logger.error("Model cannot learn from constant targets")
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
        
        # DEBUG: Check if model actually learned something
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
            max_importance = importances.max()
            num_zero_importance = (importances == 0).sum()
            
            logger.info(f"Model training completed: max_importance={max_importance:.6f}")
            logger.info(f"Features with zero importance: {num_zero_importance}/{len(importances)}")
            
            if max_importance < 1e-6:
                logger.error("CRITICAL: All feature importances are near zero!")
                logger.error("This means the model found no predictive signal - will cause flat predictions!")
                
            elif num_zero_importance > len(importances) * 0.8:
                logger.warning(f"Most features ({num_zero_importance}/{len(importances)}) have zero importance")
                logger.warning("Model may be underfitting or features lack predictive power")
            
            # Show top features
            if len(importances) > 0:
                feature_importance_pairs = list(zip(X_train.columns, importances))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                logger.info("Top 5 most important features:")
                for i, (feature, importance) in enumerate(feature_importance_pairs[:5]):
                    logger.info(f"  {i+1}. {feature}: {importance:.6f}")
        
        # Make a quick prediction check
        try:
            train_predictions = model.predict(X_train.head(100))
            pred_std = np.std(train_predictions)
            pred_range = np.max(train_predictions) - np.min(train_predictions)
            
            logger.info(f"Training predictions check: std={pred_std:.6f}, range={pred_range:.6f}")
            
            if pred_std < 1e-4:
                logger.error("CRITICAL: Model predictions on training data are nearly constant!")
                logger.error(f"Sample predictions: {train_predictions[:10].tolist()}")
                logger.error("This confirms the model is producing flat predictions!")
                
        except Exception as e:
            logger.warning(f"Could not check training predictions: {e}")
        
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
        
        # Include datetime and trading_eligible from original data for plotting
        data_file = f"{training_data_title}.csv"
        original_df = pd.read_csv(settings.get_processed_data_file(data_file))
        
        if 'datetime' in original_df.columns:
            # Get the datetime values for the test set indices
            test_df['datetime'] = original_df.loc[X_test.index, 'datetime']
            logger.info("Included datetime column in test data for plot x-axis")
        
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