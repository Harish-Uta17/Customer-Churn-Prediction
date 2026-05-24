"""
Model Training Module

Handles model training with various algorithms and hyperparameter optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from src.utils.logger import get_logger
from src.utils.constants import (
    RANDOM_STATE, LR_HYPERPARAMS, GB_HYPERPARAMS, PRIMARY_METRIC
)

logger = get_logger(__name__)


class ModelTrainer:
    """Handle model training and hyperparameter tuning."""
    
    @staticmethod
    def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
        """
        Train Logistic Regression model (our best model).
        
        Args:
            X_train: Training features (should be scaled)
            y_train: Training target
        
        Returns:
            Trained LogisticRegression model
        
        Example:
            >>> model = ModelTrainer.train_logistic_regression(X_train_scaled, y_train)
        """
        logger.info("Training Logistic Regression model...")
        
        model = LogisticRegression(**LR_HYPERPARAMS)
        model.fit(X_train, y_train)
        
        logger.info("✅ Logistic Regression trained successfully")
        logger.info(f"  Hyperparameters: {LR_HYPERPARAMS}")
        
        return model
    
    @staticmethod
    def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting model.
        
        Args:
            X_train: Training features
            y_train: Training target
        
        Returns:
            Trained GradientBoostingClassifier model
        """
        logger.info("Training Gradient Boosting model...")
        
        model = GradientBoostingClassifier(**GB_HYPERPARAMS)
        model.fit(X_train, y_train)
        
        logger.info("✅ Gradient Boosting trained successfully")
        logger.info(f"  Hyperparameters: {GB_HYPERPARAMS}")
        
        return model
    
    @staticmethod
    def train_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)
        
        logger.info("✅ Random Forest trained successfully")
        
        return model
    
    @staticmethod
    def train_multiple_models(
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train multiple models for comparison.
        
        Args:
            X_train: Training features
            y_train: Training target
        
        Returns:
            Dictionary with model names as keys and trained models as values
        """
        logger.info("Training multiple models for comparison...")
        
        models = {
            'Logistic Regression': ModelTrainer.train_logistic_regression(X_train, y_train),
            'Gradient Boosting': ModelTrainer.train_gradient_boosting(X_train, y_train),
            'Random Forest': ModelTrainer.train_random_forest(X_train, y_train),
            'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
            'XGBoost': XGBClassifier(random_state=RANDOM_STATE, verbosity=0)
        }
        
        # Train AdaBoost and XGBoost
        models['AdaBoost'].fit(X_train, y_train)
        models['XGBoost'].fit(X_train, y_train)
        
        logger.info(f"✅ All {len(models)} models trained successfully")
        
        return models
    
    @staticmethod
    def hyperparameter_tuning(
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, list],
        cv: int = 5,
        n_iter: int = 20,
        scoring: str = PRIMARY_METRIC
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            model_name: Name of model to tune
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            n_iter: Number of parameter combinations to try
            scoring: Metric to optimize
        
        Returns:
            Tuple of (best_model, best_params)
        
        Example:
            >>> param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
            >>> best_model, best_params = ModelTrainer.hyperparameter_tuning(
            ...     'Logistic Regression', X_train, y_train, param_grid
            ... )
        """
        logger.info(f"Tuning {model_name} hyperparameters...")
        logger.info(f"  CV folds: {cv}")
        logger.info(f"  Iterations: {n_iter}")
        logger.info(f"  Scoring metric: {scoring}")
        
        # Get base model
        if model_name == 'Logistic Regression':
            base_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=500)
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        else:
            logger.error(f"Unknown model: {model_name}")
            raise ValueError(f"Unknown model: {model_name}")
        
        # Hyperparameter tuning
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        logger.info(f"✅ Tuning completed")
        logger.info(f"  Best {scoring}: {search.best_score_:.4f}")
        logger.info(f"  Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_
    
    @staticmethod
    def cross_validate_model(
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        scoring: str = PRIMARY_METRIC
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training target
            cv: Number of cross-validation folds
            scoring: Metric to evaluate
        
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        
        results = {
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        logger.info(f"✅ Cross-validation completed")
        logger.info(f"  Mean {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results


# Convenience function
def train_best_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train the best performing model (Logistic Regression).
    
    Args:
        X_train: Training features (should be scaled)
        y_train: Training target
    
    Returns:
        Trained Logistic Regression model
    
    Example:
        >>> from src.models.train import train_best_model
        >>> model = train_best_model(X_train_scaled, y_train)
    """
    return ModelTrainer.train_logistic_regression(X_train, y_train)
