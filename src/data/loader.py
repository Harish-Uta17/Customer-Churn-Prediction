"""
Data Loading Module

Handles loading data from various sources (CSV, databases, APIs, etc.)
with proper error handling and logging.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.logger import get_logger
from src.utils.helpers import check_data_quality

logger = get_logger(__name__)


class DataLoader:
    """Load data from various sources."""
    
    @staticmethod
    def load_csv(
        filepath: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments to pass to pd.read_csv()
        
        Returns:
            Loaded dataframe
        
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If CSV is empty
        
        Example:
            >>> df = DataLoader.load_csv('data/churn.csv')
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            logger.info(f"Loading data from: {filepath}")
            df = pd.read_csv(filepath, **kwargs)
            logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Log data quality metrics
            quality = check_data_quality(df)
            
            return df
        
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> pd.DataFrame:
        """
        Load data from dictionary (useful for testing).
        
        Args:
            data: Dictionary with column names as keys
        
        Returns:
            Dataframe created from dictionary
        
        Example:
            >>> data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
            >>> df = DataLoader.load_from_dict(data)
        """
        try:
            logger.info("Creating dataframe from dictionary")
            df = pd.DataFrame(data)
            logger.info(f"✅ Dataframe created: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error creating dataframe from dictionary: {str(e)}")
            raise
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """
        Validate that dataframe meets basic requirements.
        
        Args:
            df: Dataframe to validate
        
        Returns:
            True if valid
        
        Raises:
            ValueError: If validation fails
        """
        if df is None or df.empty:
            raise ValueError("Dataframe is None or empty")
        if len(df) == 0:
            raise ValueError("Dataframe has no rows")
        if len(df.columns) == 0:
            raise ValueError("Dataframe has no columns")
        
        logger.info("Dataframe validation passed")
        return True
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed information about dataframe.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dictionary with info about dataframe
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        logger.info(f"Data Info:\n{pd.Series(info)}")
        return info


# Convenience function
def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
    
    Returns:
        Loaded dataframe
    
    Example:
        >>> from src.data.loader import load_data
        >>> df = load_data('data/churn.csv')
    """
    return DataLoader.load_csv(filepath, **kwargs)
