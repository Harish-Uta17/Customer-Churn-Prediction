"""
Main Entry Point for Training Pipeline

This is the main script to run the complete ML pipeline:
1. Load data
2. Clean data
3. Preprocess features
4. Split into train/test
5. Handle imbalance
6. Scale features
7. Train models
8. Evaluate models
9. Save best model

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

# Import our modules
from src.utils.logger import get_logger
from src.utils.constants import RANDOM_STATE, TARGET_COLUMN
from src.config import load_config
from src.data.loader import load_data
from src.data.cleaner import clean_data, DataCleaner, save_cleaned_data
from src.preprocessing.preprocessor import FeatureEngineer, encode_and_prepare
from src.models.train import ModelTrainer
from src.models.evaluate import ModelEvaluator
from src.models.model_manager import save_trained_model

logger = get_logger(__name__)


def main():
    """
    Run the complete ML pipeline.
    """
    logger.info("Starting Customer Churn Prediction Pipeline...")
    
    # ===== STEP 1: LOAD CONFIGURATION =====
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING CONFIGURATION")
    logger.info("="*80)
    
    config = load_config(str(PROJECT_ROOT / 'config' / 'config.yaml'))
    logger.info(f"Configuration loaded successfully")
    
    # ===== STEP 2: LOAD DATA =====
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOADING DATA")
    logger.info("="*80)
    
    raw_data_path = Path(config.get('data.raw_path', 'data/raw/churn.csv'))
    if not raw_data_path.is_absolute():
        raw_data_path = PROJECT_ROOT / raw_data_path

    df = load_data(str(raw_data_path))
    logger.info(f"Original dataset shape: {df.shape}")
    
    # ===== STEP 3: DATA CLEANING =====
    logger.info("\n" + "="*80)
    logger.info("STEP 3: DATA CLEANING")
    logger.info("="*80)
    
    processed_data_path = Path(config.get('data.processed_path', 'data/processed/churn_processed.csv'))
    if not processed_data_path.is_absolute():
        processed_data_path = PROJECT_ROOT / processed_data_path

    df = clean_data(df)
    save_cleaned_data(df, str(processed_data_path))
    logger.info(f"Cleaned dataset shape: {df.shape}")
    
    # ===== STEP 4: FEATURE ENGINEERING =====
    logger.info("\n" + "="*80)
    logger.info("STEP 4: FEATURE ENGINEERING")
    logger.info("="*80)
    
    # Encode target variable
    df, target_mapping = DataCleaner.encode_target(df, TARGET_COLUMN)
    
    # Encode categorical features
    df = encode_and_prepare(df)
    logger.info(f"After encoding: {df.shape}")
    
    # ===== STEP 5: PREPARE FEATURES =====
    logger.info("\n" + "="*80)
    logger.info("STEP 5: PREPARE FEATURES AND TARGET")
    logger.info("="*80)
    
    X, y = FeatureEngineer.prepare_features(df, TARGET_COLUMN)
    feature_names = X.columns.tolist()
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Target: {y.name}")
    
    # ===== STEP 6: TRAIN/TEST SPLIT =====
    logger.info("\n" + "="*80)
    logger.info("STEP 6: TRAIN/TEST SPLIT")
    logger.info("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get('data.test_size', 0.2),
        random_state=config.get('data.random_state', RANDOM_STATE),
        stratify=y
    )
    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    
    # ===== STEP 7: HANDLE CLASS IMBALANCE =====
    logger.info("\n" + "="*80)
    logger.info("STEP 7: HANDLE CLASS IMBALANCE (SMOTE)")
    logger.info("="*80)
    
    if config.get('preprocessing.handle_imbalance', True):
        X_train_balanced, y_train_balanced = FeatureEngineer.handle_class_imbalance(
            X_train, y_train
        )
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # ===== STEP 8: SCALE FEATURES =====
    logger.info("\n" + "="*80)
    logger.info("STEP 8: SCALE FEATURES")
    logger.info("="*80)
    
    X_train_scaled, X_test_scaled, scaler = FeatureEngineer.scale_features(
        X_train_balanced, X_test,
        method=config.get('preprocessing.scaling_method', 'standard'),
        return_scaler=True
    )
    
    # ===== STEP 9: TRAIN MODELS =====
    logger.info("\n" + "="*80)
    logger.info("STEP 9: TRAINING MODELS")
    logger.info("="*80)
    
    models = ModelTrainer.train_multiple_models(X_train_scaled, y_train_balanced)
    
    # ===== STEP 10: EVALUATE MODELS =====
    logger.info("\n" + "="*80)
    logger.info("STEP 10: EVALUATING MODELS")
    logger.info("="*80)
    
    comparison_df = ModelEvaluator.compare_models(models, X_test_scaled, y_test)
    
    # ===== STEP 11: SELECT AND SAVE BEST MODEL =====
    logger.info("\n" + "="*80)
    logger.info("STEP 11: SAVING BEST MODEL")
    logger.info("="*80)
    
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = models[best_model_name]
    best_metrics = comparison_df.iloc[0].to_dict()
    
    model_path = save_trained_model(
        best_model,
        scaler,
        feature_names,
        best_metrics,
        model_name=f"churn_model_best"
    )
    
    logger.info(f"Best model saved: {model_path}")
    logger.info(f"Model: {best_model_name}")
    logger.info(f"Metrics: {best_metrics}")
    
    # ===== PIPELINE COMPLETE =====
    logger.info("\n" + "="*80)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"ROC-AUC: {best_metrics.get('ROC-AUC', 'N/A'):.4f}")
    logger.info(f"Saved to: {model_path}")
    
    return {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': best_metrics,
        'model_path': model_path
    }


if __name__ == "__main__":
    try:
        result = main()
        logger.info("\nPipeline finished successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise
