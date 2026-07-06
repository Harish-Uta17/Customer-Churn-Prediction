"""
Helper Functions and Utilities

Reusable utility functions used across the project.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across numpy, pandas, and sklearn.
    
    Args:
        seed: Random seed value
    
    Example:
        >>> set_random_seed(42)
        >>> # Now all random operations are reproducible
    """
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed} for reproducibility")


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check data quality and return a report.
    
    Args:
        df: Input dataframe
    
    Returns:
        Dictionary with quality metrics
    
    Example:
        >>> quality_report = check_data_quality(df)
        >>> print(f"Missing values: {quality_report['missing_count']}")
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_count': int(df.isnull().sum().sum()) if df is not None and len(df) > 0 and len(df.columns) > 0 else 0,
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if df is not None and len(df) > 0 and len(df.columns) > 0 else 0.0,
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    logger.info(f"Data quality report: {quality_report}")
    return quality_report


def validate_train_test_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> bool:
    """
    Validate that train/test split is correct.
    
    Args:
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If validation fails
    """
    # Check shapes match
    if X_train.shape[0] != len(y_train):
        raise ValueError("X_train and y_train length mismatch")
    if X_test.shape[0] != len(y_test):
        raise ValueError("X_test and y_test length mismatch")
    
    # Check columns match
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError("X_train and X_test have different columns")
    
    # Check no overlap
    if X_train.index.intersection(X_test.index).size > 0:
        logger.warning("Train and test sets have overlapping indices")
    
    logger.info(f"Train/test split validation passed")
    logger.info(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    return True


def get_class_distribution(y: pd.Series) -> Dict[str, int]:
    """
    Get class distribution in target variable.
    
    Args:
        y: Target series
    
    Returns:
        Dictionary with class counts and percentages
    """
    counts = y.value_counts()
    class_0 = int(counts.get(0, 0))
    class_1 = int(counts.get(1, 0))
    imbalance_ratio = None
    try:
        small = min([v for v in counts.tolist() if v > 0]) if not counts.empty else 0
        if small and max(counts.tolist()) > 0:
            imbalance_ratio = max(counts.tolist()) / small
    except Exception:
        imbalance_ratio = None

    distribution = {
        'class_0': class_0,
        'class_1': class_1,
        'imbalance_ratio': imbalance_ratio
    }
    
    logger.info(f"Class distribution: {distribution}")
    return distribution


def compare_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name1: str = "Dataset 1",
    name2: str = "Dataset 2"
) -> None:
    """
    Compare two datasets and print differences.
    
    Args:
        df1, df2: Dataframes to compare
        name1, name2: Dataset names for logging
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparing {name1} vs {name2}")
    logger.info(f"{'='*60}")
    logger.info(f"{name1}: {df1.shape}")
    logger.info(f"{name2}: {df2.shape}")
    logger.info(f"Missing values {name1}: {df1.isnull().sum().sum()}")
    logger.info(f"Missing values {name2}: {df2.isnull().sum().sum()}")
    logger.info(f"Duplicate rows {name1}: {df1.duplicated().sum()}")
    logger.info(f"Duplicate rows {name2}: {df2.duplicated().sum()}")
