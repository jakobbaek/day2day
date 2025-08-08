"""Model implementations for day2day."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from .base import BaseModel

# Import model libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """XGBoost regression model."""
    
    def __init__(self, name: str = "xgboost", **kwargs):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")
        
        super().__init__(name, **kwargs)
        
        # Default parameters
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        self.model = xgb.XGBRegressor(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train XGBoost model."""
        self.feature_names = list(X_train.columns)
        
        # XGBoost supports sample weights
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples and exponential decay weights")
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples")
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost."""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_trained:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class RandomForestModel(BaseModel):
    """Random Forest regression model."""
    
    def __init__(self, name: str = "random_forest", **kwargs):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed")
        
        super().__init__(name, **kwargs)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42
        }
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        self.model = RandomForestRegressor(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train Random Forest model."""
        self.feature_names = list(X_train.columns)
        
        # RandomForest supports sample weights
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples and exponential decay weights")
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples")
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with Random Forest."""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_trained:
            return None
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


class LinearRegressionModel(BaseModel):
    """Linear regression model."""
    
    def __init__(self, name: str = "linear_regression", **kwargs):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed")
        
        super().__init__(name, **kwargs)
        self.model = LinearRegression(**kwargs)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train linear regression model."""
        self.feature_names = list(X_train.columns)
        
        # LinearRegression supports sample weights
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples and exponential decay weights")
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples")
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with linear regression."""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (coefficients)."""
        if not self.is_trained:
            return None
        
        coef = self.model.coef_
        return dict(zip(self.feature_names, coef))


class RidgeRegressionModel(BaseModel):
    """Ridge regression model."""
    
    def __init__(self, name: str = "ridge_regression", **kwargs):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed")
        
        super().__init__(name, **kwargs)
        
        # Default parameters
        default_params = {
            'alpha': 1.0,
            'random_state': 42
        }
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        self.model = Ridge(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train ridge regression model."""
        self.feature_names = list(X_train.columns)
        
        # Ridge supports sample weights
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples and exponential decay weights")
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples")
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with ridge regression."""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (coefficients)."""
        if not self.is_trained:
            return None
        
        coef = self.model.coef_
        return dict(zip(self.feature_names, coef))


class LassoRegressionModel(BaseModel):
    """Lasso regression model."""
    
    def __init__(self, name: str = "lasso_regression", **kwargs):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed")
        
        super().__init__(name, **kwargs)
        
        # Default parameters
        default_params = {
            'alpha': 1.0,
            'random_state': 42
        }
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        self.model = Lasso(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train lasso regression model."""
        self.feature_names = list(X_train.columns)
        
        # Lasso supports sample weights
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples and exponential decay weights")
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples")
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with lasso regression."""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (coefficients)."""
        if not self.is_trained:
            return None
        
        coef = self.model.coef_
        return dict(zip(self.feature_names, coef))


class SVRModel(BaseModel):
    """Support Vector Regression model."""
    
    def __init__(self, name: str = "svr", **kwargs):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed")
        
        super().__init__(name, **kwargs)
        
        # Default parameters
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        }
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        self.model = SVR(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train SVR model."""
        self.feature_names = list(X_train.columns)
        
        # SVR supports sample weights
        if sample_weight is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples and exponential decay weights")
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples")
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with SVR."""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """SVR doesn't provide feature importance."""
        return None


class MLPModel(BaseModel):
    """Multi-layer Perceptron regression model."""
    
    def __init__(self, name: str = "mlp", **kwargs):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn not installed")
        
        super().__init__(name, **kwargs)
        
        # Default parameters
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 200,
            'random_state': 42
        }
        
        # Merge with user parameters
        params = {**default_params, **kwargs}
        self.model = MLPRegressor(**params)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[np.ndarray] = None) -> None:
        """Train MLP model."""
        self.feature_names = list(X_train.columns)
        
        # MLPRegressor does NOT support sample weights - warn if provided
        if sample_weight is not None:
            logger.warning(f"MLPRegressor does not support sample weights - training {self.name} without exponential decay")
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples (weights ignored)")
        else:
            self.model.fit(X_train, y_train)
            logger.info(f"Trained {self.name} model with {len(X_train)} samples")
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with MLP."""
        X = self.validate_input(X)
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """MLP doesn't provide feature importance."""
        return None


# Model registry
MODEL_REGISTRY = {
    'xgboost': XGBoostModel,
    'random_forest': RandomForestModel,
    'linear_regression': LinearRegressionModel,
    'ridge_regression': RidgeRegressionModel,
    'lasso_regression': LassoRegressionModel,
    'svr': SVRModel,
    'mlp': MLPModel
}

# Models that can handle null values natively (important for day trading)
NULL_COMPATIBLE_MODELS = {
    'xgboost': XGBoostModel,
    'random_forest': RandomForestModel
}


def get_available_models() -> Dict[str, type]:
    """Get dictionary of available model types."""
    available = {}
    
    for name, model_class in MODEL_REGISTRY.items():
        try:
            # Try to instantiate to check if dependencies are available
            model_class(name="test")
            available[name] = model_class
        except ImportError:
            continue
    
    return available


def get_null_compatible_models() -> Dict[str, type]:
    """
    Get dictionary of models that can handle null values natively.
    
    This is important for day trading where lagged features may have null values
    at market open (first few minutes have no recent history from previous day).
    
    Returns:
        Dictionary of null-compatible model types
    """
    available = {}
    
    for name, model_class in NULL_COMPATIBLE_MODELS.items():
        try:
            # Try to instantiate to check if dependencies are available
            model_class(name="test")
            available[name] = model_class
            logger.debug(f"Null-compatible model available: {name}")
        except ImportError:
            logger.warning(f"Null-compatible model {name} not available due to missing dependencies")
            continue
    
    return available


def create_model(model_type: str, name: str = None, **kwargs) -> BaseModel:
    """
    Create a model instance.
    
    Args:
        model_type: Type of model to create
        name: Optional name for the model
        **kwargs: Model-specific parameters
        
    Returns:
        Model instance
        
    Raises:
        ValueError: If model type is not available
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    model_name = name or model_type
    
    try:
        model = model_class(name=model_name, **kwargs)
        # Store the base model type for bootstrap file naming
        model.model_type = model_type
        return model
    except ImportError as e:
        raise ValueError(f"Model {model_type} not available: {e}")