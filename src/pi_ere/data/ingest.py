"""Main data ingestion orchestrator for PI-ERE.

This module coordinates data collection from multiple OSINT sources and manages
the ingestion pipeline.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from pi_ere.utils.config import config


class DataSource(ABC):
    """Abstract base class for all data sources.

    Each data source connector must implement this interface.
    """

    def __init__(self, name: str, source_config: Optional[Dict[str, Any]] = None):
        """Initialize data source.

        Args:
            name: Name of the data source (e.g., 'acled', 'gdelt')
            source_config: Source-specific configuration dictionary
        """
        self.name = name
        self.config = source_config or config.data_sources.get(name, {})
        self.enabled = self.config.get('enabled', True)
        self.update_frequency = self.config.get('update_frequency', 'daily')

        # Setup paths
        self.raw_data_dir = config.data_dir / 'raw' / name
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized {name} data source (enabled={self.enabled})")

    @abstractmethod
    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        regions: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data from the source for specified date range and regions.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            regions: List of region/country identifiers (ISO codes or names)
            **kwargs: Additional source-specific parameters

        Returns:
            DataFrame with fetched data, must include columns:
                - date: datetime
                - region: str
                - source: str (data source name)
                - Additional source-specific columns
        """
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Validate fetched data quality.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes, False otherwise
        """
        pass

    def save_raw(self, df: pd.DataFrame, filename: Optional[str] = None) -> Path:
        """Save raw data to disk.

        Args:
            df: DataFrame to save
            filename: Optional custom filename, defaults to timestamped name

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.name}_{timestamp}.parquet"

        filepath = self.raw_data_dir / filename
        df.to_parquet(filepath, compression='snappy', index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")

        return filepath

    def load_raw(self, filename: str) -> pd.DataFrame:
        """Load raw data from disk.

        Args:
            filename: Name of file to load

        Returns:
            DataFrame with loaded data
        """
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")

        return df

    def get_latest_file(self) -> Optional[Path]:
        """Get the most recently saved raw data file.

        Returns:
            Path to latest file, or None if no files exist
        """
        files = sorted(self.raw_data_dir.glob(f"{self.name}_*.parquet"))
        return files[-1] if files else None


class DataIngestionOrchestrator:
    """Orchestrates data ingestion from multiple sources.

    This class manages the overall ingestion pipeline, coordinating multiple
    data sources and handling scheduling, retries, and error handling.
    """

    def __init__(self):
        """Initialize the orchestrator."""
        self.sources: Dict[str, DataSource] = {}
        self.config = config

        logger.info("Initialized DataIngestionOrchestrator")

    def register_source(self, source: DataSource):
        """Register a data source with the orchestrator.

        Args:
            source: DataSource instance to register
        """
        if source.enabled:
            self.sources[source.name] = source
            logger.info(f"Registered data source: {source.name}")
        else:
            logger.warning(f"Skipped disabled data source: {source.name}")

    def ingest_all(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        regions: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        save_raw: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Ingest data from all registered sources.

        Args:
            start_date: Start date for ingestion (string or datetime)
            end_date: End date for ingestion (string or datetime)
            regions: List of regions to fetch (None = all available)
            sources: List of source names to ingest (None = all registered)
            save_raw: Whether to save raw data to disk

        Returns:
            Dictionary mapping source names to DataFrames
        """
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        logger.info(
            f"Starting ingestion: {start_date.date()} to {end_date.date()}, "
            f"regions={regions}, sources={sources or 'all'}"
        )

        results = {}
        sources_to_ingest = sources or list(self.sources.keys())

        for source_name in sources_to_ingest:
            if source_name not in self.sources:
                logger.warning(f"Source '{source_name}' not registered, skipping")
                continue

            source = self.sources[source_name]
            logger.info(f"Fetching from {source_name}...")

            try:
                df = source.fetch(
                    start_date=start_date,
                    end_date=end_date,
                    regions=regions
                )

                # Validate
                if not source.validate(df):
                    logger.error(f"Validation failed for {source_name}")
                    continue

                # Save if requested
                if save_raw and not df.empty:
                    source.save_raw(df)

                results[source_name] = df
                logger.info(
                    f"Successfully ingested {len(df)} records from {source_name}"
                )

            except Exception as e:
                logger.error(f"Error ingesting from {source_name}: {e}")
                logger.exception(e)
                continue

        logger.info(
            f"Ingestion complete. Fetched from {len(results)}/{len(sources_to_ingest)} sources"
        )

        return results

    def ingest_incremental(
        self,
        lookback_days: int = 7,
        regions: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Perform incremental ingestion (fetch recent data only).

        Args:
            lookback_days: Number of days to look back from today
            regions: List of regions to fetch
            sources: List of source names to ingest

        Returns:
            Dictionary mapping source names to DataFrames
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        logger.info(f"Starting incremental ingestion ({lookback_days} days)")

        return self.ingest_all(
            start_date=start_date,
            end_date=end_date,
            regions=regions,
            sources=sources
        )

    def get_ingestion_status(self) -> pd.DataFrame:
        """Get status of all registered data sources.

        Returns:
            DataFrame with source status information
        """
        status_data = []

        for name, source in self.sources.items():
            latest_file = source.get_latest_file()

            status_data.append({
                'source': name,
                'enabled': source.enabled,
                'update_frequency': source.update_frequency,
                'latest_file': latest_file.name if latest_file else None,
                'latest_file_date': (
                    datetime.fromtimestamp(latest_file.stat().st_mtime)
                    if latest_file else None
                ),
            })

        return pd.DataFrame(status_data)


def get_default_regions() -> List[str]:
    """Get default list of regions for extractive industries.

    Returns:
        List of ISO country codes for key extractive regions
    """
    return [
        # Africa
        'COD',  # Democratic Republic of Congo
        'ZMB',  # Zambia
        'ZAF',  # South Africa
        'GHA',  # Ghana
        'MLI',  # Mali
        'TZA',  # Tanzania
        'MOZ',  # Mozambique
        'NGA',  # Nigeria
        'AGO',  # Angola

        # Middle East
        'IRQ',  # Iraq
        'SAU',  # Saudi Arabia
        'ARE',  # UAE

        # Central Asia
        'KAZ',  # Kazakhstan
        'UZB',  # Uzbekistan
        'TKM',  # Turkmenistan

        # South Asia
        'AFG',  # Afghanistan
        'PAK',  # Pakistan
        'IND',  # India

        # Latin America
        'VEN',  # Venezuela
        'COL',  # Colombia
        'PER',  # Peru
        'CHL',  # Chile
        'BRA',  # Brazil
        'MEX',  # Mexico
    ]
