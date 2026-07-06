from pathlib import Path

import pandas as pd
import numpy as np
from pandas.api.types import is_object_dtype, is_string_dtype
from typing import List, Optional, Tuple
from src.utils.logger import get_logger
from src.utils.constants import DROP_COLUMNS, TARGET_COLUMN, HANDLE_MISSING

logger = get_logger(__name__)


class DataCleaner:    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = HANDLE_MISSING,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        
        
        
        df_copy = df.copy()
        
        missing_count = df_copy.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No missing values found")
            return df_copy
        
        logger.info(f"Found {missing_count} missing values")
        
        if columns is None:
            columns = df_copy.columns[df_copy.isnull().any()].tolist()
        
        if strategy == 'median':
            for col in columns:
                if df_copy[col].isnull().sum() > 0:
                    median_val = df_copy[col].median()
                    df_copy[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled {col} with median: {median_val}")
        
        elif strategy == 'mean':
            for col in columns:
                if df_copy[col].isnull().sum() > 0:
                    mean_val = df_copy[col].mean()
                    df_copy[col].fillna(mean_val, inplace=True)
                    logger.info(f"Filled {col} with mean: {mean_val}")
        
        elif strategy == 'drop':
            df_copy = df_copy.dropna(subset=columns)
            logger.info(f"Dropped rows with missing values")
        
        elif strategy == 'forward_fill':
            df_copy[columns] = df_copy[columns].ffill()
            logger.info(f"Used forward fill for missing values")
        
        else:
            logger.warning(f"Unknown strategy: {strategy}, returning unchanged")
            return df
        
        logger.info(f"✅ Missing values handled: {df_copy.isnull().sum().sum()} remaining")
        return df_copy
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from dataframe.
        
        Args:
            df: Input dataframe
            subset: Columns to consider for duplicates (None = all)
        
        Returns:
            Dataframe with duplicates removed
        """
        df_copy = df.copy()
        
        duplicates_before = df_copy.duplicated().sum()
        logger.info(f"Found {duplicates_before} duplicate rows")
        
        if duplicates_before == 0:
            logger.info("No duplicates found")
            return df_copy
        
        df_copy = df_copy.drop_duplicates(subset=subset)
        logger.info(f"✅ Duplicates removed: {duplicates_before} rows dropped")
        
        return df_copy
    
    @staticmethod
    def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix data types in dataframe.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with corrected data types
        
        Example:
            >>> df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        """
        df_copy = df.copy()
        
        # Try to convert numeric-looking text columns to numeric values.
        for col in df_copy.columns:
            if is_object_dtype(df_copy[col]) or is_string_dtype(df_copy[col]):
                try:
                    numeric_col = pd.to_numeric(df_copy[col], errors='coerce')
                    non_null_ratio = numeric_col.notna().mean()
                    if non_null_ratio >= 0.9:
                        if numeric_col.isna().any():
                            fill_value = numeric_col.median()
                            numeric_col = numeric_col.fillna(fill_value)
                            logger.info(f"Filled missing values in {col} with median: {fill_value}")
                        df_copy[col] = numeric_col
                        logger.info(f"Converted {col} to numeric")
                except Exception:
                    pass  # Keep as text
        
        # Convert boolean strings to int
        bool_cols = df_copy.select_dtypes(include='bool').columns
        for col in bool_cols:
            df_copy[col] = df_copy[col].astype(int)
            logger.info(f"Converted {col} to int")
        
        logger.info(f"✅ Data types fixed")
        logger.info(f"\nData types:\n{df_copy.dtypes}")
        
        return df_copy
    
    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Drop specified columns from dataframe.
        
        Args:
            df: Input dataframe
            columns: List of column names to drop
        
        Returns:
            Dataframe with columns dropped
        
        Example:
            >>> df = DataCleaner.drop_columns(df, ['customerID', 'unwanted_col'])
        """
        df_copy = df.copy()
        
        # Only drop columns that exist
        columns_to_drop = [col for col in columns if col in df_copy.columns]
        
        if not columns_to_drop:
            logger.info("No columns to drop")
            return df_copy
        
        df_copy = df_copy.drop(columns=columns_to_drop)
        logger.info(f"✅ Dropped columns: {columns_to_drop}")
        logger.info(f"  Remaining columns: {df_copy.shape[1]}")
        
        return df_copy
    
    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers using z-score method.
        
        Args:
            df: Input dataframe
            columns: Numeric columns to check for outliers (None = all numeric)
            threshold: Z-score threshold (default 3.0 = 99.7% of data)
        
        Returns:
            Dataframe with outliers removed
        
        Example:
            >>> df = DataCleaner.remove_outliers(df, threshold=3.0)
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            logger.info("No numeric columns found")
            return df_copy
        
        from scipy import stats
        z_scores = np.abs(stats.zscore(df_copy[columns], nan_policy='omit'))
        outlier_mask = (z_scores < threshold).all(axis=1)
        
        rows_before = len(df_copy)
        df_copy = df_copy[outlier_mask]
        rows_removed = rows_before - len(df_copy)
        
        logger.info(f"✅ Outliers removed: {rows_removed} rows ({rows_removed/rows_before*100:.2f}%)")
        
        return df_copy
    
    @staticmethod
    def encode_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, dict]:
        """
        Encode target variable to numeric.
        
        Args:
            df: Input dataframe
            target_col: Name of target column
        
        Returns:
            Tuple of (encoded dataframe, encoding mapping)
        
        Example:
            >>> df, mapping = DataCleaner.encode_target(df, 'Churn')
            >>> # mapping = {'No': 0, 'Yes': 1}
        """
        df_copy = df.copy()
        
        if target_col not in df_copy.columns:
            logger.error(f"Target column {target_col} not found")
            raise ValueError(f"Target column {target_col} not found")
        
        if df_copy[target_col].dtype in ['int64', 'float64']:
            logger.info(f"Target column {target_col} is already numeric")
            return df_copy, {}
        
        # Create mapping
        unique_values = df_copy[target_col].unique()
        encoding_map = {val: idx for idx, val in enumerate(sorted(unique_values))}
        
        df_copy[target_col] = df_copy[target_col].map(encoding_map)
        
        logger.info(f"✅ Target {target_col} encoded: {encoding_map}")
        
        return df_copy, encoding_map


# Convenience functions
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full data cleaning pipeline.
    
    Args:
        df: Input dataframe
    
    Returns:
        Cleaned dataframe
    
    Example:
        >>> from src.data.cleaner import clean_data
        >>> df = clean_data(df)
    """
    df = DataCleaner.handle_missing_values(df)
    df = DataCleaner.remove_duplicates(df)
    df = DataCleaner.fix_data_types(df)
    df = DataCleaner.drop_columns(df, DROP_COLUMNS)
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> str:
    """Save cleaned dataframe to disk and return the saved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"✅ Cleaned data saved to: {path}")
    return str(path)
