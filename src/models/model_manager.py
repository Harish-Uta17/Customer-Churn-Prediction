"""
Model Manager

Handles saving and loading trained models, preprocessors, and metadata.
Ensures reproducibility and production readiness.
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelManager:
    """Manage model persistence and loading."""
    
    @staticmethod
    def save_model(
        model: Any,
        scaler: StandardScaler,
        preprocessor: Dict[str, Any],
        metadata: Dict[str, Any],
        model_name: str = "churn_model",
        models_dir: str = "models"
    ) -> str:
        """
        Save trained model, scaler, preprocessor, and metadata.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            preprocessor: Preprocessing information
            metadata: Model metadata (metrics, hyperparams, etc.)
            model_name: Name of the model
            models_dir: Directory to save model
        
        Returns:
            Path to saved model
        
        Example:
            >>> path = ModelManager.save_model(
            ...     model, scaler, preprocessor,
            ...     {'roc_auc': 0.8363, 'trained_date': datetime.now().isoformat()},
            ...     model_name='lr_v1'
            ... )
        """
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        # Create model artifacts dictionary
        artifacts = {
            'model': model,
            'scaler': scaler,
            'preprocessor': preprocessor,
            'metadata': metadata
        }
        
        # Save as pickle
        model_file = models_path / f"{model_name}.pkl"
        
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(artifacts, f)
            
            logger.info(f"✅ Model saved: {model_file}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
        
        # Save metadata as JSON for readability
        metadata_file = models_path / f"{model_name}_metadata.json"
        
        try:
            # Convert datetime objects to strings for JSON serialization
            metadata_serializable = {}
            for key, value in metadata.items():
                if isinstance(value, datetime):
                    metadata_serializable[key] = value.isoformat()
                elif isinstance(value, np.ndarray):
                    metadata_serializable[key] = value.tolist()
                else:
                    metadata_serializable[key] = value
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_serializable, f, indent=2)
            
            logger.info(f"✅ Metadata saved: {metadata_file}")
        except Exception as e:
            logger.warning(f"Warning: Could not save metadata JSON: {str(e)}")
        
        return str(model_file)
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[Any, StandardScaler, Dict[str, Any], Dict[str, Any]]:
        """
        Load trained model, scaler, preprocessor, and metadata.
        
        Args:
            model_path: Path to saved model pickle file
        
        Returns:
            Tuple of (model, scaler, preprocessor, metadata)
        
        Raises:
            FileNotFoundError: If model file doesn't exist
        
        Example:
            >>> model, scaler, preprocessor, metadata = ModelManager.load_model('models/lr_v1.pkl')
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            logger.info(f"Loading model from: {model_path}")
            
            with open(model_path, 'rb') as f:
                artifacts = pickle.load(f)
            
            model = artifacts['model']
            scaler = artifacts['scaler']
            preprocessor = artifacts.get('preprocessor', {})
            metadata = artifacts.get('metadata', {})
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"  Model type: {type(model).__name__}")
            logger.info(f"  Metadata: {metadata}")
            
            return model, scaler, preprocessor, metadata
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @staticmethod
    def list_available_models(models_dir: str = "models") -> list:
        """
        List all available saved models.
        
        Args:
            models_dir: Directory containing models
        
        Returns:
            List of model names
        """
        models_path = Path(models_dir)
        
        if not models_path.exists():
            logger.info(f"Models directory does not exist: {models_path}")
            return []
        
        model_files = list(models_path.glob("*.pkl"))
        model_names = [f.stem for f in model_files]
        
        logger.info(f"Available models ({len(model_names)}):")
        for name in model_names:
            logger.info(f"  - {name}")
        
        return model_names
    
    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """
        Get information about a saved model.
        
        Args:
            model_path: Path to saved model
        
        Returns:
            Dictionary with model information
        """
        model, scaler, preprocessor, metadata = ModelManager.load_model(model_path)
        
        info = {
            'model_path': model_path,
            'model_type': type(model).__name__,
            'metadata': metadata,
            'scaler_type': type(scaler).__name__,
            'file_size_kb': Path(model_path).stat().st_size / 1024
        }
        
        return info


# Convenience function
def save_trained_model(
    model: Any,
    scaler: StandardScaler,
    feature_names: list,
    metrics: Dict[str, Any],
    model_name: str = "best_model"
) -> str:
    """
    Save trained model with all necessary artifacts.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names used during training
        metrics: Evaluation metrics
        model_name: Name to save model as
    
    Returns:
        Path to saved model file
    
    Example:
        >>> from src.models.model_manager import save_trained_model
        >>> path = save_trained_model(
        ...     model, scaler, feature_names,
        ...     {'roc_auc': 0.8363},
        ...     'logistic_regression_v1'
        ... )
    """
    preprocessor = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'scaling_method': 'StandardScaler'
    }
    
    metadata = {
        'model_type': type(model).__name__,
        'trained_date': datetime.now(),
        'metrics': metrics,
        'hyperparameters': getattr(model, 'get_params', lambda: {})()
    }
    
    return ModelManager.save_model(
        model, scaler, preprocessor, metadata,
        model_name=model_name
    )
