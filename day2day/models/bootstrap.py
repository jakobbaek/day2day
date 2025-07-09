"""Bootstrapping functionality for model uncertainty estimation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import json
import logging
from pathlib import Path
from .base import BaseModel
from .implementations import create_model
from ..config.settings import settings
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BootstrapEstimator:
    """Handles bootstrap sampling and uncertainty estimation."""
    
    def __init__(self, n_bootstrap: int = 100, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.bootstrap_results: List[Dict[str, Any]] = []
        
    def bootstrap_sample(self, X: pd.DataFrame, y: pd.Series, 
                        replace: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create a bootstrap sample from the data.
        
        Args:
            X: Features
            y: Target
            replace: Whether to sample with replacement
            
        Returns:
            Tuple of (X_bootstrap, y_bootstrap)
        """
        n_samples = len(X)
        
        # Generate bootstrap indices
        np.random.seed(self.random_state)
        indices = np.random.choice(n_samples, size=n_samples, replace=replace)
        
        X_bootstrap = X.iloc[indices].reset_index(drop=True)
        y_bootstrap = y.iloc[indices].reset_index(drop=True)
        
        return X_bootstrap, y_bootstrap
    
    def train_bootstrap_models(self, 
                             model_type: str,
                             model_params: Dict[str, Any],
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Train multiple bootstrap models.
        
        Args:
            model_type: Type of model to train
            model_params: Model parameters
            X_train: Training features
            y_train: Training target
            X_test: Test features for prediction
            
        Returns:
            List of bootstrap results
        """
        bootstrap_results = []
        
        logger.info(f"Training {self.n_bootstrap} bootstrap models of type {model_type}")
        
        for i in range(self.n_bootstrap):
            try:
                # Create bootstrap sample
                X_boot, y_boot = self.bootstrap_sample(X_train, y_train)
                
                # Train model
                model = create_model(model_type, name=f"bootstrap_{i}", **model_params)
                model.train(X_boot, y_boot)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Store results
                result = {
                    'bootstrap_id': i,
                    'predictions': predictions,
                    'feature_importance': model.get_feature_importance(),
                    'model_type': model_type
                }
                
                bootstrap_results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{self.n_bootstrap} bootstrap models")
                    
            except Exception as e:
                logger.error(f"Bootstrap {i} failed: {e}")
                continue
        
        logger.info(f"Successfully trained {len(bootstrap_results)} bootstrap models")
        return bootstrap_results
    
    def calculate_prediction_intervals(self, 
                                     bootstrap_predictions: np.ndarray,
                                     confidence_levels: List[float] = [0.68, 0.95]) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals from bootstrap predictions.
        
        Args:
            bootstrap_predictions: Array of shape (n_bootstrap, n_samples)
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary of prediction intervals
        """
        intervals = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
            
            intervals[f"{confidence:.0%}"] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        return intervals
    
    def estimate_prediction_probability(self, 
                                      bootstrap_predictions: np.ndarray,
                                      threshold: float,
                                      direction: str = 'above') -> np.ndarray:
        """
        Estimate probability that predictions exceed a threshold.
        
        Args:
            bootstrap_predictions: Array of shape (n_bootstrap, n_samples)
            threshold: Threshold value
            direction: 'above' or 'below'
            
        Returns:
            Array of probabilities for each sample
        """
        if direction == 'above':
            exceed_threshold = bootstrap_predictions > threshold
        else:
            exceed_threshold = bootstrap_predictions < threshold
        
        probabilities = np.mean(exceed_threshold, axis=0)
        return probabilities
    
    def create_ensemble_prediction(self, 
                                 bootstrap_predictions: np.ndarray,
                                 method: str = 'mean') -> np.ndarray:
        """
        Create ensemble prediction from bootstrap results.
        
        Args:
            bootstrap_predictions: Array of shape (n_bootstrap, n_samples)
            method: Aggregation method ('mean', 'median', 'trimmed_mean')
            
        Returns:
            Ensemble predictions
        """
        if method == 'mean':
            return np.mean(bootstrap_predictions, axis=0)
        elif method == 'median':
            return np.median(bootstrap_predictions, axis=0)
        elif method == 'trimmed_mean':
            # Remove top and bottom 10% before taking mean
            sorted_preds = np.sort(bootstrap_predictions, axis=0)
            trim_size = int(0.1 * len(sorted_preds))
            if trim_size > 0:
                trimmed_preds = sorted_preds[trim_size:-trim_size]
            else:
                trimmed_preds = sorted_preds
            return np.mean(trimmed_preds, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def run_bootstrap_analysis(self, 
                             training_data_title: str,
                             target_instrument: str,
                             model_type: str,
                             model_params: Dict[str, Any],
                             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Run complete bootstrap analysis.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_type: Type of model to bootstrap
            model_params: Model parameters
            test_size: Test set size
            
        Returns:
            Bootstrap analysis results
        """
        logger.info(f"Starting bootstrap analysis for {training_data_title}_{target_instrument}")
        
        # Load training data
        data_file = f"{training_data_title}.csv"
        file_path = settings.get_processed_data_file(data_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Training data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Prepare features and target
        exclude_cols = ['datetime', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['target']
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Create train/test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Run bootstrap training
        bootstrap_results = self.train_bootstrap_models(
            model_type, model_params, X_train, y_train, X_test
        )
        
        # Extract predictions
        bootstrap_predictions = np.array([result['predictions'] for result in bootstrap_results])
        
        # Calculate statistics
        ensemble_prediction = self.create_ensemble_prediction(bootstrap_predictions)
        prediction_intervals = self.calculate_prediction_intervals(bootstrap_predictions)
        
        # Calculate prediction uncertainty
        prediction_std = np.std(bootstrap_predictions, axis=0)
        
        # Aggregate feature importance
        feature_importances = [result['feature_importance'] for result in bootstrap_results 
                             if result['feature_importance'] is not None]
        
        avg_feature_importance = {}
        if feature_importances:
            all_features = feature_importances[0].keys()
            for feature in all_features:
                importances = [fi[feature] for fi in feature_importances if feature in fi]
                avg_feature_importance[feature] = {
                    'mean': np.mean(importances),
                    'std': np.std(importances)
                }
        
        # Compile results
        results = {
            'model_type': model_type,
            'n_bootstrap': len(bootstrap_results),
            'ensemble_prediction': ensemble_prediction,
            'prediction_intervals': prediction_intervals,
            'prediction_std': prediction_std,
            'feature_importance': avg_feature_importance,
            'bootstrap_predictions': bootstrap_predictions,
            'actual_values': y_test.values,
            'test_indices': X_test.index.tolist()
        }
        
        # Save results
        self.save_bootstrap_results(training_data_title, target_instrument, model_type, results)
        
        return results
    
    def save_bootstrap_results(self, 
                             training_data_title: str,
                             target_instrument: str,
                             model_type: str,
                             results: Dict[str, Any]) -> None:
        """
        Save bootstrap analysis results.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_type: Model type
            results: Bootstrap results
        """
        suite_name = f"{training_data_title}_{target_instrument}"
        suite_dir = settings.models_path / suite_name
        suite_dir.mkdir(exist_ok=True)
        
        # Save ensemble prediction and intervals
        bootstrap_file = suite_dir / f"bootstrap_{model_type}.npz"
        
        np.savez(
            bootstrap_file,
            ensemble_prediction=results['ensemble_prediction'],
            prediction_intervals=results['prediction_intervals'],
            prediction_std=results['prediction_std'],
            bootstrap_predictions=results['bootstrap_predictions'],
            actual_values=results['actual_values'],
            test_indices=results['test_indices']
        )
        
        # Save metadata and feature importance
        metadata = {
            'model_type': model_type,
            'n_bootstrap': results['n_bootstrap'],
            'feature_importance': results['feature_importance'],
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = suite_dir / f"bootstrap_{model_type}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved bootstrap results to {bootstrap_file}")
    
    def load_bootstrap_results(self, 
                             training_data_title: str,
                             target_instrument: str,
                             model_type: str) -> Dict[str, Any]:
        """
        Load bootstrap analysis results.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_type: Model type
            
        Returns:
            Bootstrap results
        """
        suite_name = f"{training_data_title}_{target_instrument}"
        suite_dir = settings.models_path / suite_name
        
        bootstrap_file = suite_dir / f"bootstrap_{model_type}.npz"
        metadata_file = suite_dir / f"bootstrap_{model_type}_metadata.json"
        
        if not bootstrap_file.exists():
            raise FileNotFoundError(f"Bootstrap results not found: {bootstrap_file}")
        
        # Load numerical results
        data = np.load(bootstrap_file, allow_pickle=True)
        
        # Load metadata
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        results = {
            'ensemble_prediction': data['ensemble_prediction'],
            'prediction_intervals': data['prediction_intervals'].item(),
            'prediction_std': data['prediction_std'],
            'bootstrap_predictions': data['bootstrap_predictions'],
            'actual_values': data['actual_values'],
            'test_indices': data['test_indices'],
            **metadata
        }
        
        return results
    
    def get_probability_above_threshold(self, 
                                      training_data_title: str,
                                      target_instrument: str,
                                      model_type: str,
                                      threshold: float) -> np.ndarray:
        """
        Get probability that predictions exceed threshold.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_type: Model type
            threshold: Threshold value
            
        Returns:
            Array of probabilities
        """
        results = self.load_bootstrap_results(training_data_title, target_instrument, model_type)
        bootstrap_predictions = results['bootstrap_predictions']
        
        return self.estimate_prediction_probability(bootstrap_predictions, threshold, 'above')
    
    def get_confidence_interval(self, 
                              training_data_title: str,
                              target_instrument: str,
                              model_type: str,
                              confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Get confidence interval for predictions.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_type: Model type
            confidence_level: Confidence level
            
        Returns:
            Dictionary with lower and upper bounds
        """
        results = self.load_bootstrap_results(training_data_title, target_instrument, model_type)
        
        confidence_key = f"{confidence_level:.0%}"
        if confidence_key in results['prediction_intervals']:
            return results['prediction_intervals'][confidence_key]
        else:
            # Calculate on-the-fly
            bootstrap_predictions = results['bootstrap_predictions']
            intervals = self.calculate_prediction_intervals(bootstrap_predictions, [confidence_level])
            return intervals[confidence_key]