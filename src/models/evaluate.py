"""
Model Evaluation Module

Handles model evaluation using various metrics and visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
from src.utils.logger import get_logger
from src.utils.constants import EVALUATION_METRICS, PRIMARY_METRIC

logger = get_logger(__name__)


class ModelEvaluator:
    """Handle model evaluation and metrics calculation."""
    
    @staticmethod
    def evaluate_model(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set using multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of model (for logging)
        
        Returns:
            Dictionary with evaluation metrics
        
        Example:
            >>> metrics = ModelEvaluator.evaluate_model(model, X_test, y_test, "Logistic Regression")
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Add ROC-AUC if we have probabilities
        if y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"✅ Evaluation completed for {model_name}")
        for metric, value in metrics.items():
            if metric != 'Model':
                logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    @staticmethod
    def compare_models(
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate and compare multiple models.
        
        Args:
            models: Dictionary of model_name: model pairs
            X_test: Test features
            y_test: Test target
        
        Returns:
            DataFrame with comparison of all models
        
        Example:
            >>> comparison_df = ModelEvaluator.compare_models(models, X_test, y_test)
        """
        logger.info(f"Comparing {len(models)} models...")
        
        results = []
        for name, model in models.items():
            metrics = ModelEvaluator.evaluate_model(model, X_test, y_test, name)
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        sort_column = 'ROC-AUC' if 'ROC-AUC' in results_df.columns else PRIMARY_METRIC
        results_df = results_df.sort_values(sort_column, ascending=False)
        
        logger.info(f"\n{'='*80}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*80}")
        logger.info(f"\n{results_df.to_string(index=False)}\n")
        
        return results_df
    
    @staticmethod
    def get_confusion_matrix(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> np.ndarray:
        """
        Get confusion matrix for model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        
        Returns:
            Confusion matrix
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"Confusion Matrix:")
        logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return cm
    
    @staticmethod
    def get_roc_curve(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get ROC curve data for model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        
        Returns:
            Tuple of (fpr, tpr, auc_score)
        """
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        if y_pred_proba is None:
            logger.warning("Model does not support predict_proba, cannot compute ROC curve")
            return None, None, None
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        logger.info(f"ROC Curve computed (AUC = {auc_score:.4f})")
        
        return fpr, tpr, auc_score
    
    @staticmethod
    def get_classification_report(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        
        Returns:
            Classification report as string
        """
        from sklearn.metrics import classification_report
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
        
        logger.info(f"\nClassification Report:\n{report}")
        
        return report
    
    @staticmethod
    def get_feature_importance(model: Any, feature_names: list, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        # Check if model has feature importance
        if hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_[0])
            logger.info(f"Using model coefficients (Linear model)")
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
            logger.info(f"Using model feature importances (Tree-based model)")
        else:
            logger.warning(f"Model does not have feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(top_n)
        
        logger.info(f"\nTop {top_n} Features:")
        logger.info(importance_df.to_string(index=False))
        
        return importance_df


# Convenience function
def evaluate_best_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluate the best model and return all metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with all evaluation metrics
    
    Example:
        >>> from src.models.evaluate import evaluate_best_model
        >>> metrics = evaluate_best_model(model, X_test, y_test)
    """
    metrics = ModelEvaluator.evaluate_model(model, X_test, y_test, "Best Model")
    cm = ModelEvaluator.get_confusion_matrix(model, X_test, y_test)
    report = ModelEvaluator.get_classification_report(model, X_test, y_test)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report
    }
