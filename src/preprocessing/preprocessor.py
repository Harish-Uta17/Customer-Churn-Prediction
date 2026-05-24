"""
Preprocessing and Feature Engineering Module

Handles feature engineering, encoding, scaling, and SMOTE for class imbalance.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger
from src.utils.constants import TARGET_COLUMN

logger = get_logger(__name__)


class FeatureEngineer:
    """Handle feature engineering and preprocessing."""
    
    @staticmethod
    def encode_categorical_features(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using one-hot encoding.
        
        Args:
            df: Input dataframe
            drop_first: Whether to drop first category (default True for model compatibility)
        
        Returns:
            Dataframe with encoded features
        
        Example:
            >>> df = FeatureEngineer.encode_categorical_features(df)
        """
        df_copy = df.copy()
        
        # Identify categorical columns
        categorical_cols = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column if it's in categorical columns
        if TARGET_COLUMN in categorical_cols:
            categorical_cols.remove(TARGET_COLUMN)
        
        if not categorical_cols:
            logger.info("No categorical features to encode")
            return df_copy
        
        logger.info(f"Encoding {len(categorical_cols)} categorical features: {categorical_cols}")
        
        # One-hot encoding
        df_copy = pd.get_dummies(df_copy, columns=categorical_cols, drop_first=drop_first)
        
        logger.info(f"✅ Categorical encoding completed")
        logger.info(f"  Features increased from {len(df.columns)} to {len(df_copy.columns)}")
        
        return df_copy
    
    @staticmethod
    def convert_boolean_to_int(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert boolean columns to integers (0/1).
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with boolean columns converted to int
        """
        df_copy = df.copy()
        
        bool_cols = df_copy.select_dtypes(include='bool').columns
        
        if len(bool_cols) == 0:
            logger.info("No boolean columns found")
            return df_copy
        
        for col in bool_cols:
            df_copy[col] = df_copy[col].astype(int)
        
        logger.info(f"✅ Converted {len(bool_cols)} boolean columns to int")
        
        return df_copy
    
    @staticmethod
    def scale_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        method: str = 'standard',
        return_scaler: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[object]]:
        """
        Scale features using StandardScaler or MinMaxScaler.
        
        IMPORTANT: Fit scaler only on training data, then transform both train and test!
        
        Args:
            X_train: Training features
            X_test: Testing features
            method: 'standard' or 'minmax'
            return_scaler: Whether to return the fitted scaler
        
        Returns:
            Tuple of (scaled_X_train, scaled_X_test) or with scaler if return_scaler=True
        
        Example:
            >>> X_train_scaled, X_test_scaled, scaler = FeatureEngineer.scale_features(
            ...     X_train, X_test, method='standard', return_scaler=True
            ... )
        """
        logger.info(f"Scaling features using {method} scaler")
        logger.info(f"  IMPORTANT: Fitting scaler on training data only")
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.error(f"Unknown scaling method: {method}")
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit on training data only (never on test data!)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Only transform, never fit on test
        
        logger.info(f"✅ Features scaled successfully")
        logger.info(f"  Train shape: {X_train_scaled.shape}")
        logger.info(f"  Test shape: {X_test_scaled.shape}")
        
        if return_scaler:
            return X_train_scaled, X_test_scaled, scaler
        else:
            return X_train_scaled, X_test_scaled, None
    
    @staticmethod
    def handle_class_imbalance(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'smote',
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE or other methods.
        
        IMPORTANT: Apply SMOTE ONLY on training data, never on test data!
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'smote' (only option currently)
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (balanced X_train, balanced y_train)
        
        Example:
            >>> X_train_balanced, y_train_balanced = FeatureEngineer.handle_class_imbalance(
            ...     X_train, y_train
            ... )
        """
        # Check class distribution before
        logger.info("Class distribution BEFORE balancing:")
        logger.info(y_train.value_counts().to_dict())
        
        if method == 'smote':
            logger.info(f"Applying SMOTE to balance training data...")
            smote = SMOTE(random_state=random_state)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            
            logger.info("Class distribution AFTER balancing:")
            logger.info(pd.Series(y_balanced).value_counts().to_dict())
            logger.info(f"✅ SMOTE applied successfully")
            logger.info(f"  Samples: {len(X_train)} → {len(X_balanced)}")
            
            return X_balanced, y_balanced
        
        else:
            logger.error(f"Unknown balancing method: {method}")
            raise ValueError(f"Unknown balancing method: {method}")
    
    @staticmethod
    def prepare_features(
        df: pd.DataFrame,
        target_col: str = TARGET_COLUMN
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from dataframe.
        
        Args:
            df: Input dataframe
            target_col: Name of target column
        
        Returns:
            Tuple of (X features, y target)
        
        Example:
            >>> X, y = FeatureEngineer.prepare_features(df)
        """
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            raise ValueError(f"Target column {target_col} not found")
        
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        logger.info(f"Features prepared")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  y shape: {y.shape}")
        
        return X, y


# Convenience functions for common workflows
def encode_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply categorical encoding and boolean conversion in sequence.
    
    Args:
        df: Input dataframe
    
    Returns:
        Processed dataframe
    """
    df = FeatureEngineer.encode_categorical_features(df)
    df = FeatureEngineer.convert_boolean_to_int(df)
    return df
