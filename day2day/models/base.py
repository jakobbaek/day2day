"""Base classes for model implementations."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import joblib
from pathlib import Path


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.config = kwargs
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        pass
    
    def save(self, filepath: Path) -> None:
        """Save model to file."""
        model_data = {
            'name': self.name,
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'config': self.config,
            'model_type': getattr(self, 'model_type', None)
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: Path) -> None:
        """Load model from file."""
        model_data = joblib.load(filepath)
        self.name = model_data['name']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        # Load model_type if available (for bootstrap file naming compatibility)
        self.model_type = model_data.get('model_type', None)
    
    def validate_input(self, X: pd.DataFrame) -> None:
        """Validate input data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            extra_features = set(X.columns) - set(self.feature_names)
            if extra_features:
                X = X[self.feature_names]
        
        return X


class ModelEnsemble:
    """Ensemble of multiple models."""
    
    def __init__(self, models: Dict[str, BaseModel], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or {name: 1.0 for name in models.keys()}
        self.is_trained = False
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train all models in the ensemble."""
        for model in self.models.values():
            model.train(X_train, y_train, sample_weight=sample_weight)
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights[name]
            predictions.append(pred * weight)
            total_weight += weight
        
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0) / total_weight
        return ensemble_pred
    
    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models."""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions