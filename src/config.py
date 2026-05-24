"""
Configuration Management

Loads and manages configuration from YAML files.
Supports environment-specific configs (dev, production, etc.)
"""

import yaml
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Load and manage configuration from YAML files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize ConfigManager with a config file.
        
        Args:
            config_path: Path to the YAML configuration file
        
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from: {self.config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {str(e)}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        Supports nested keys with dot notation: 'model.hyperparameters.C'
        
        Args:
            key: Configuration key (supports nested access with dots)
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        
        Example:
            >>> config = ConfigManager()
            >>> c_value = config.get('model.hyperparameters.C', default=1.0)
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value if value is not None else default
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to configuration."""
        return self.get(key)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager({self.config_path})"


def load_config(config_path: str = "config/config.yaml") -> ConfigManager:
    """
    Load configuration from file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)
