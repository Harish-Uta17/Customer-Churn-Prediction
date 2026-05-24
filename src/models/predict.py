"""
Model Prediction Module

Handles making predictions on new data using trained models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List
from src.utils.logger import get_logger
from src.utils.constants import CHURN_RISK_THRESHOLD

logger = get_logger(__name__)


class ModelPredictor:
    """Handle model predictions on new data."""
    
    @staticmethod
    def predict_single(
        model: Any,
        X_single: Union[np.ndarray, pd.DataFrame],
        return_probability: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction for a single customer.
        
        Args:
            model: Trained model
            X_single: Single sample (1, n_features)
            return_probability: Whether to return probabilities
        
        Returns:
            Dictionary with prediction and probability
        
        Example:
            >>> prediction = ModelPredictor.predict_single(model, customer_data)
            >>> print(prediction)
            {'prediction': 1, 'probability': 0.75, 'risk_level': 'HIGH'}
        """
        # Ensure input is 2D
        if isinstance(X_single, pd.DataFrame):
            X_single = X_single.values
        
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X_single)[0]
        
        result = {'prediction': int(prediction)}
        
        # Get probability if available
        if return_probability and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_single)[0]
            result['probability_no_churn'] = float(proba[0])
            result['probability_churn'] = float(proba[1])
            result['confidence'] = float(max(proba))
            
            # Determine risk level
            churn_prob = proba[1]
            if churn_prob >= 0.7:
                result['risk_level'] = 'HIGH'
            elif churn_prob >= 0.4:
                result['risk_level'] = 'MEDIUM'
            else:
                result['risk_level'] = 'LOW'
        
        return result
    
    @staticmethod
    def predict_batch(
        model: Any,
        X_batch: Union[np.ndarray, pd.DataFrame],
        return_probability: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for batch of customers.
        
        Args:
            model: Trained model
            X_batch: Batch of samples (n_samples, n_features)
            return_probability: Whether to return probabilities
        
        Returns:
            DataFrame with predictions
        
        Example:
            >>> predictions_df = ModelPredictor.predict_batch(model, X_test)
        """
        if isinstance(X_batch, pd.DataFrame):
            X_batch = X_batch.values
        
        logger.info(f"Making predictions for {len(X_batch)} samples...")
        
        # Make predictions
        predictions = model.predict(X_batch)
        
        results = pd.DataFrame({
            'prediction': predictions,
            'churn': predictions  # 1 = will churn, 0 = won't churn
        })
        
        # Get probabilities if available
        if return_probability and hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_batch)
            results['probability_no_churn'] = proba[:, 0]
            results['probability_churn'] = proba[:, 1]
            results['confidence'] = results[['probability_no_churn', 'probability_churn']].max(axis=1)
            
            # Determine risk levels
            def get_risk_level(prob):
                if prob >= 0.7:
                    return 'HIGH'
                elif prob >= 0.4:
                    return 'MEDIUM'
                else:
                    return 'LOW'
            
            results['risk_level'] = results['probability_churn'].apply(get_risk_level)
        
        logger.info(f"✅ Predictions completed")
        logger.info(f"  Churn predictions: {results['churn'].sum()} out of {len(results)}")
        
        return results
    
    @staticmethod
    def identify_high_risk_customers(
        predictions_df: pd.DataFrame,
        threshold: float = CHURN_RISK_THRESHOLD
    ) -> pd.DataFrame:
        """
        Identify high-risk customers from predictions.
        
        Args:
            predictions_df: DataFrame with predictions
            threshold: Probability threshold for high-risk
        
        Returns:
            DataFrame with high-risk customers
        
        Example:
            >>> high_risk = ModelPredictor.identify_high_risk_customers(predictions_df)
        """
        high_risk = predictions_df[predictions_df['probability_churn'] >= threshold]
        
        logger.info(f"High-risk customers identified: {len(high_risk)}")
        
        return high_risk
    
    @staticmethod
    def get_prediction_summary(predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of predictions.
        
        Args:
            predictions_df: DataFrame with predictions
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_predictions': len(predictions_df),
            'churn_predictions': predictions_df['churn'].sum(),
            'no_churn_predictions': (predictions_df['churn'] == 0).sum(),
            'churn_percentage': (predictions_df['churn'].sum() / len(predictions_df)) * 100,
            'avg_churn_probability': predictions_df['probability_churn'].mean(),
            'high_risk_count': (predictions_df['probability_churn'] >= 0.7).sum(),
            'medium_risk_count': ((predictions_df['probability_churn'] >= 0.4) & (predictions_df['probability_churn'] < 0.7)).sum(),
            'low_risk_count': (predictions_df['probability_churn'] < 0.4).sum()
        }
        
        logger.info(f"\nPrediction Summary:")
        logger.info(f"  Total Predictions: {summary['total_predictions']}")
        logger.info(f"  Churn: {summary['churn_predictions']} ({summary['churn_percentage']:.2f}%)")
        logger.info(f"  High Risk: {summary['high_risk_count']}")
        logger.info(f"  Medium Risk: {summary['medium_risk_count']}")
        logger.info(f"  Low Risk: {summary['low_risk_count']}")
        
        return summary


# Convenience function
def predict_customer_churn(model: Any, customer_data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
    """
    Predict churn for one or more customers.
    
    Args:
        model: Trained model
        customer_data: Single customer or batch of customers
    
    Returns:
        Prediction result(s)
    
    Example:
        >>> from src.models.predict import predict_customer_churn
        >>> result = predict_customer_churn(model, customer_data)
    """
    if isinstance(customer_data, np.ndarray):
        if customer_data.ndim == 1:
            return ModelPredictor.predict_single(model, customer_data)
        else:
            return ModelPredictor.predict_batch(model, customer_data)
    else:  # DataFrame
        if len(customer_data) == 1:
            return ModelPredictor.predict_single(model, customer_data)
        else:
            return ModelPredictor.predict_batch(model, customer_data)
