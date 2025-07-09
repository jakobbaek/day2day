"""Model evaluation and visualization module."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import json
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .training import ModelTrainer
from ..config.settings import settings

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')


class ModelEvaluator:
    """Handles model evaluation and visualization."""
    
    def __init__(self):
        self.settings = settings
        self.trainer = ModelTrainer()
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf,
            'directional_accuracy': np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))
        }
        
        return metrics
    
    def evaluate_model(self, 
                      training_data_title: str,
                      target_instrument: str,
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_name: Name of the model to evaluate
            
        Returns:
            Evaluation results
        """
        # Load model and test data
        models = self.trainer.load_model_suite(training_data_title, target_instrument)
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found in suite")
        
        model = models[model_name]
        
        # Load test data
        suite_name = f"{training_data_title}_{target_instrument}"
        suite_dir = settings.models_path / suite_name
        test_file = suite_dir / "test_data.csv"
        
        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        
        test_df = pd.read_csv(test_file)
        
        # Separate features and target
        X_test = test_df.drop(['target'], axis=1)
        y_test = test_df['target']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test.values, y_pred)
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'actual': y_test.values,
            'test_size': len(y_test)
        }
        
        return results
    
    def evaluate_model_suite(self, 
                           training_data_title: str,
                           target_instrument: str) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all models in a suite.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            
        Returns:
            Dictionary of evaluation results for each model
        """
        logger.info(f"Evaluating model suite: {training_data_title}_{target_instrument}")
        
        # Load models
        models = self.trainer.load_model_suite(training_data_title, target_instrument)
        
        results = {}
        
        for model_name in models.keys():
            try:
                model_results = self.evaluate_model(training_data_title, target_instrument, model_name)
                results[model_name] = model_results
                logger.info(f"Evaluated {model_name}: RMSE={model_results['metrics']['rmse']:.4f}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Save evaluation results
        self.save_evaluation_results(training_data_title, target_instrument, results)
        
        return results
    
    def save_evaluation_results(self, 
                              training_data_title: str,
                              target_instrument: str,
                              results: Dict[str, Dict[str, Any]]) -> None:
        """
        Save evaluation results.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            results: Evaluation results
        """
        suite_name = f"{training_data_title}_{target_instrument}"
        suite_dir = settings.models_path / suite_name
        
        # Save metrics summary
        metrics_summary = {}
        for model_name, model_results in results.items():
            metrics_summary[model_name] = model_results['metrics']
        
        metrics_file = suite_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2, default=str)
        
        # Save detailed results
        detailed_file = suite_dir / "evaluation_detailed.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                'model_name': model_results['model_name'],
                'metrics': model_results['metrics'],
                'feature_importance': model_results['feature_importance'],
                'test_size': model_results['test_size']
            }
        
        with open(detailed_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Saved evaluation results to {suite_dir}")
    
    def create_prediction_plot(self, 
                             training_data_title: str,
                             target_instrument: str,
                             model_names: List[str] = None,
                             save_plot: bool = True) -> plt.Figure:
        """
        Create prediction vs actual plot.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_names: List of model names to plot (None for all)
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Load evaluation results
        results = self.evaluate_model_suite(training_data_title, target_instrument)
        
        if model_names is None:
            model_names = list(results.keys())
        
        # Create figure
        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            if model_name not in results:
                continue
                
            model_results = results[model_name]
            y_true = model_results['actual']
            y_pred = model_results['predictions']
            
            # Create timeline plot
            ax = axes[i]
            
            # Skip weekends and create continuous timeline
            df = pd.DataFrame({
                'actual': y_true,
                'predicted': y_pred,
                'index': range(len(y_true))
            })
            
            ax.plot(df['index'], df['actual'], label='Actual', color='blue', linewidth=1.5)
            ax.plot(df['index'], df['predicted'], label='Predicted', color='orange', 
                   linestyle='--', linewidth=1.5)
            
            ax.set_title(f'{model_name} - RMSE: {model_results["metrics"]["rmse"]:.4f}')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            suite_name = f"{training_data_title}_{target_instrument}"
            suite_dir = settings.models_path / suite_name
            plot_file = suite_dir / "prediction_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction plot to {plot_file}")
        
        return fig
    
    def create_scatter_plot(self, 
                          training_data_title: str,
                          target_instrument: str,
                          model_names: List[str] = None,
                          save_plot: bool = True) -> plt.Figure:
        """
        Create scatter plot of predictions vs actual.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_names: List of model names to plot (None for all)
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Load evaluation results
        results = self.evaluate_model_suite(training_data_title, target_instrument)
        
        if model_names is None:
            model_names = list(results.keys())
        
        # Create figure
        n_models = len(model_names)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        elif cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, model_name in enumerate(model_names):
            if model_name not in results:
                continue
                
            model_results = results[model_name]
            y_true = model_results['actual']
            y_pred = model_results['predictions']
            
            ax = axes[i]
            
            # Create scatter plot
            ax.scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model_name} - R²: {model_results["metrics"]["r2"]:.4f}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(model_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            suite_name = f"{training_data_title}_{target_instrument}"
            suite_dir = settings.models_path / suite_name
            plot_file = suite_dir / "scatter_plot.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved scatter plot to {plot_file}")
        
        return fig
    
    def create_feature_importance_plot(self, 
                                     training_data_title: str,
                                     target_instrument: str,
                                     model_name: str,
                                     top_n: int = 20,
                                     save_plot: bool = True) -> plt.Figure:
        """
        Create feature importance plot.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            model_name: Name of the model
            top_n: Number of top features to show
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Evaluate model
        results = self.evaluate_model(training_data_title, target_instrument, model_name)
        
        feature_importance = results['feature_importance']
        
        if feature_importance is None:
            raise ValueError(f"Model {model_name} doesn't provide feature importance")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N features
        top_features = sorted_features[:top_n]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.3)))
        
        features = [f[0] for f in top_features]
        importances = [f[1] for f in top_features]
        
        y_pos = np.arange(len(features))
        
        ax.barh(y_pos, importances)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Feature Importance - {model_name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            suite_name = f"{training_data_title}_{target_instrument}"
            suite_dir = settings.models_path / suite_name
            plot_file = suite_dir / f"feature_importance_{model_name}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {plot_file}")
        
        return fig
    
    def create_metrics_comparison(self, 
                                training_data_title: str,
                                target_instrument: str,
                                save_plot: bool = True) -> plt.Figure:
        """
        Create metrics comparison plot.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            save_plot: Whether to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Load evaluation results
        results = self.evaluate_model_suite(training_data_title, target_instrument)
        
        # Prepare data for plotting
        model_names = list(results.keys())
        metrics_data = []
        
        for model_name in model_names:
            metrics = results[model_name]['metrics']
            metrics_data.append({
                'model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape'] if metrics['mape'] != np.inf else 0
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # RMSE
        axes[0, 0].bar(df['model'], df['RMSE'])
        axes[0, 0].set_title('Root Mean Squared Error (RMSE)')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE
        axes[0, 1].bar(df['model'], df['MAE'])
        axes[0, 1].set_title('Mean Absolute Error (MAE)')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R²
        axes[1, 0].bar(df['model'], df['R²'])
        axes[1, 0].set_title('R² Score')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE
        axes[1, 1].bar(df['model'], df['MAPE'])
        axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plot:
            suite_name = f"{training_data_title}_{target_instrument}"
            suite_dir = settings.models_path / suite_name
            plot_file = suite_dir / "metrics_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics comparison plot to {plot_file}")
        
        return fig
    
    def get_best_model(self, 
                      training_data_title: str,
                      target_instrument: str,
                      metric: str = 'rmse') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best model based on a metric.
        
        Args:
            training_data_title: Title of the training data
            target_instrument: Target instrument
            metric: Metric to optimize ('rmse', 'mae', 'r2', 'mape')
            
        Returns:
            Tuple of (best_model_name, best_model_results)
        """
        results = self.evaluate_model_suite(training_data_title, target_instrument)
        
        if metric in ['rmse', 'mae', 'mape']:
            # Lower is better
            best_model = min(results.items(), key=lambda x: x[1]['metrics'][metric])
        elif metric == 'r2':
            # Higher is better
            best_model = max(results.items(), key=lambda x: x[1]['metrics'][metric])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_model