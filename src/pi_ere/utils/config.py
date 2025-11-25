"""Configuration management for PI-ERE."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Config:
    """Central configuration manager for PI-ERE.

    Loads configuration from YAML files and environment variables.
    Provides easy access to configuration values throughout the application.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration.

        Args:
            config_dir: Path to configuration directory. If None, uses project root/config.
        """
        if config_dir is None:
            # Assume we're in src/pi_ere/utils, go up to project root
            config_dir = Path(__file__).parent.parent.parent.parent / "config"

        self.config_dir = Path(config_dir)
        self._data_sources: Optional[Dict[str, Any]] = None
        self._model_config: Optional[Dict[str, Any]] = None
        self._env_vars = self._load_env_vars()

    def _load_env_vars(self) -> Dict[str, str]:
        """Load relevant environment variables."""
        env_vars = {}

        # Data source API keys
        env_vars['ACLED_API_KEY'] = os.getenv('ACLED_API_KEY', '')
        env_vars['GDELT_API_KEY'] = os.getenv('GDELT_API_KEY', '')
        env_vars['WORLD_BANK_API_KEY'] = os.getenv('WORLD_BANK_API_KEY', '')

        # Database connections
        env_vars['POSTGRES_URL'] = os.getenv('POSTGRES_URL', '')
        env_vars['QDRANT_URL'] = os.getenv('QDRANT_URL', 'http://localhost:6333')
        env_vars['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY', '')

        # Paths
        env_vars['DATA_DIR'] = os.getenv('DATA_DIR', 'data')
        env_vars['MODEL_DIR'] = os.getenv('MODEL_DIR', 'models')

        # Model settings
        env_vars['DEVICE'] = os.getenv('DEVICE', 'cpu')  # 'cpu' or 'cuda'

        return env_vars

    @property
    def data_sources(self) -> Dict[str, Any]:
        """Load and cache data sources configuration."""
        if self._data_sources is None:
            config_path = self.config_dir / "data_sources.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self._data_sources = yaml.safe_load(f)
            else:
                self._data_sources = self._get_default_data_sources()
        return self._data_sources

    @property
    def model_config(self) -> Dict[str, Any]:
        """Load and cache model configuration."""
        if self._model_config is None:
            config_path = self.config_dir / "model_config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self._model_config = yaml.safe_load(f)
            else:
                self._model_config = self._get_default_model_config()
        return self._model_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation, e.g., 'model.embedding.dim')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # First check environment variables
        if key.upper() in self._env_vars:
            return self._env_vars[key.upper()]

        # Then check configuration files
        parts = key.split('.')

        # Try model config first
        config = self.model_config
        try:
            for part in parts:
                config = config[part]
            return config
        except (KeyError, TypeError):
            pass

        # Try data sources
        config = self.data_sources
        try:
            for part in parts:
                config = config[part]
            return config
        except (KeyError, TypeError):
            pass

        return default

    def _get_default_data_sources(self) -> Dict[str, Any]:
        """Return default data sources configuration."""
        return {
            'acled': {
                'enabled': True,
                'api_key_env': 'ACLED_API_KEY',
                'base_url': 'https://api.acleddata.com/acled/read',
                'update_frequency': 'weekly',
            },
            'gdelt': {
                'enabled': True,
                'base_url': 'http://data.gdeltproject.org/events',
                'update_frequency': 'daily',
            },
            'world_bank': {
                'enabled': True,
                'api_key_env': 'WORLD_BANK_API_KEY',
                'indicators': [
                    'NY.GDP.MKTP.CD',  # GDP
                    'FP.CPI.TOTL.ZG',  # Inflation
                    'SL.UEM.TOTL.ZS',  # Unemployment
                ],
                'update_frequency': 'quarterly',
            },
            'commodities': {
                'enabled': True,
                'sources': {
                    'fred': {
                        'base_url': 'https://api.stlouisfed.org/fred',
                        'series': [
                            'DCOILWTICO',  # WTI Crude Oil
                            'GOLDAMGBD228NLBM',  # Gold
                            'PCOPPUSDM',  # Copper
                        ],
                    },
                    'yahoo_finance': {
                        'tickers': ['GC=F', 'CL=F', 'HG=F'],  # Gold, Crude, Copper
                    },
                },
                'update_frequency': 'daily',
            },
        }

    def _get_default_model_config(self) -> Dict[str, Any]:
        """Return default model configuration."""
        return {
            'embedding': {
                'ts_model': 'ts2vec',
                'text_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'embedding_dim': 320,
                'window_size': 30,  # days
            },
            'forecasting': {
                'model_type': 'chronos',  # 'chronos', 'tft', 'prophet', 'arima'
                'forecast_horizon': 90,  # days (3 months)
                'prediction_length': 30,
                'context_length': 365,  # 1 year of history
                'batch_size': 32,
            },
            'anomaly': {
                'methods': ['residual', 'embedding_drift', 'isolation_forest'],
                'threshold_percentile': 95,
                'min_anomaly_duration': 3,  # days
            },
            'vector_db': {
                'backend': 'qdrant',  # 'qdrant' or 'faiss'
                'collection_name': 'pi_ere_embeddings',
                'similarity_metric': 'cosine',
                'top_k': 10,
            },
            'training': {
                'device': 'cpu',  # Will be overridden by DEVICE env var
                'epochs': 50,
                'learning_rate': 0.001,
                'early_stopping_patience': 10,
                'validation_split': 0.2,
            },
        }

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self._env_vars['DATA_DIR'])

    @property
    def model_dir(self) -> Path:
        """Get model directory path."""
        return Path(self._env_vars['MODEL_DIR'])

    @property
    def device(self) -> str:
        """Get compute device (cpu or cuda)."""
        return self._env_vars['DEVICE']


# Global config instance
config = Config()
