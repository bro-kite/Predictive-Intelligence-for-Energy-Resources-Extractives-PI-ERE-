"""Data harmonization and preprocessing for PI-ERE.

This module takes raw data from multiple sources and transforms it into a unified,
analysis-ready format suitable for embedding and modeling.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger

from pi_ere.utils.config import config


class DataHarmonizer:
    """Harmonizes data from multiple sources into unified time-series panel.

    The harmonizer:
    1. Aligns data to common time granularity (daily/weekly)
    2. Handles missing data
    3. Creates unified entity-date-feature panel
    4. Performs basic feature engineering
    5. Validates data quality
    """

    def __init__(
        self,
        granularity: str = 'D',
        timezone: str = 'UTC',
    ):
        """Initialize data harmonizer.

        Args:
            granularity: Time granularity ('D'=daily, 'W'=weekly, 'M'=monthly)
            timezone: Timezone for date alignment
        """
        self.granularity = granularity
        self.timezone = timezone
        self.config = config

        # Quality thresholds from config
        self.max_missing_ratio = config.get(
            'data_quality.max_missing_ratio',
            0.3
        )

        self.processed_data_dir = config.data_dir / 'processed'
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized DataHarmonizer (granularity={granularity}, tz={timezone})"
        )

    def harmonize(
        self,
        data_sources: Dict[str, pd.DataFrame],
        regions: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Harmonize data from multiple sources into unified panel.

        Args:
            data_sources: Dictionary mapping source names to DataFrames
            regions: List of regions to include (None = all)
            start_date: Start date for panel (None = earliest available)
            end_date: End date for panel (None = latest available)

        Returns:
            Harmonized DataFrame with columns:
                - region: str (ISO country code or region identifier)
                - date: datetime
                - feature_name: str
                - value: float
        """
        logger.info(
            f"Harmonizing {len(data_sources)} data sources "
            f"(regions={regions or 'all'})"
        )

        harmonized_dfs = []

        for source_name, df in data_sources.items():
            if df.empty:
                logger.warning(f"Skipping empty source: {source_name}")
                continue

            try:
                # Harmonize individual source
                harmonized = self._harmonize_source(
                    df,
                    source_name,
                    regions,
                    start_date,
                    end_date,
                )

                if not harmonized.empty:
                    harmonized_dfs.append(harmonized)
                    logger.info(
                        f"Harmonized {source_name}: {len(harmonized)} records"
                    )

            except Exception as e:
                logger.error(f"Error harmonizing {source_name}: {e}")
                logger.exception(e)
                continue

        if not harmonized_dfs:
            logger.error("No data sources successfully harmonized")
            return pd.DataFrame()

        # Combine all sources
        combined = pd.concat(harmonized_dfs, ignore_index=True)

        # Create full time-series panel (fill gaps)
        panel = self._create_panel(combined, regions, start_date, end_date)

        # Handle missing data
        panel = self._handle_missing_data(panel)

        logger.info(
            f"Harmonization complete: {len(panel)} records, "
            f"{panel['region'].nunique()} regions, "
            f"{panel['feature_name'].nunique()} features"
        )

        return panel

    def _harmonize_source(
        self,
        df: pd.DataFrame,
        source_name: str,
        regions: Optional[List[str]],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Harmonize a single data source.

        Args:
            df: Source DataFrame
            source_name: Name of the source
            regions: Regions to filter
            start_date: Start date
            end_date: End date

        Returns:
            Harmonized DataFrame in long format
        """
        # Ensure date column exists
        if 'date' not in df.columns:
            logger.error(f"{source_name}: No 'date' column found")
            return pd.DataFrame()

        # Convert to datetime
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_convert(self.timezone)

        # Filter by date range
        if start_date is not None:
            df = df[df['date'] >= start_date]
        if end_date is not None:
            df = df[df['date'] <= end_date]

        # Filter by regions
        if regions is not None and 'region' in df.columns:
            df = df[df['region'].isin(regions)]

        if df.empty:
            return pd.DataFrame()

        # Align to granularity
        df = self._align_to_granularity(df)

        # Convert to long format (entity-date-feature-value)
        long_df = self._to_long_format(df, source_name)

        return long_df

    def _align_to_granularity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align data to specified time granularity.

        Args:
            df: DataFrame with date column

        Returns:
            DataFrame aggregated to target granularity
        """
        # Create period based on granularity
        df['period'] = df['date'].dt.to_period(self.granularity)

        # Identify numeric columns to aggregate
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude date-related columns
        numeric_cols = [col for col in numeric_cols if col not in ['year', 'month', 'day']]

        # Group by region and period
        group_cols = ['region', 'period'] if 'region' in df.columns else ['period']

        # Aggregate
        agg_dict = {col: 'sum' for col in numeric_cols}

        # Special handling for certain columns (mean instead of sum)
        mean_cols = [
            'tone', 'avg_tone', 'goldstein_scale', 'avg_goldstein',
            'latitude', 'longitude', 'price',
        ]

        for col in mean_cols:
            if col in agg_dict:
                agg_dict[col] = 'mean'

        grouped = df.groupby(group_cols).agg(agg_dict).reset_index()

        # Convert period back to timestamp
        grouped['date'] = grouped['period'].dt.to_timestamp()
        grouped = grouped.drop('period', axis=1)

        return grouped

    def _to_long_format(
        self,
        df: pd.DataFrame,
        source_name: str
    ) -> pd.DataFrame:
        """Convert DataFrame to long format (entity-date-feature-value).

        Args:
            df: DataFrame in wide format
            source_name: Name of data source

        Returns:
            Long-format DataFrame
        """
        # Identify ID columns (entity and time)
        id_cols = ['region', 'date'] if 'region' in df.columns else ['date']

        # Identify value columns (exclude ID and metadata columns)
        exclude_cols = set(id_cols + ['source', 'data_source', 'country_name'])
        value_cols = [col for col in df.columns if col not in exclude_cols]

        if not value_cols:
            logger.warning(f"{source_name}: No value columns found")
            return pd.DataFrame()

        # Melt to long format
        long_df = df.melt(
            id_vars=id_cols,
            value_vars=value_cols,
            var_name='feature_name',
            value_name='value'
        )

        # Add source prefix to feature names
        long_df['feature_name'] = source_name + '_' + long_df['feature_name'].astype(str)

        # Convert value to numeric
        long_df['value'] = pd.to_numeric(long_df['value'], errors='coerce')

        # Remove null values
        long_df = long_df.dropna(subset=['value'])

        return long_df

    def _create_panel(
        self,
        df: pd.DataFrame,
        regions: Optional[List[str]],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
    ) -> pd.DataFrame:
        """Create complete time-series panel with all region-date combinations.

        Args:
            df: Long-format DataFrame
            regions: Regions to include
            start_date: Start date
            end_date: End date

        Returns:
            Complete panel DataFrame
        """
        # Get unique regions and dates
        unique_regions = regions if regions else df['region'].unique()
        unique_features = df['feature_name'].unique()

        # Determine date range
        min_date = start_date or df['date'].min()
        max_date = end_date or df['date'].max()

        # Create date range based on granularity
        date_range = pd.date_range(
            start=min_date,
            end=max_date,
            freq=self.granularity,
            tz=self.timezone
        )

        # Create multi-index for complete panel
        panel_index = pd.MultiIndex.from_product(
            [unique_regions, date_range, unique_features],
            names=['region', 'date', 'feature_name']
        )

        # Create empty panel
        panel_df = pd.DataFrame(index=panel_index).reset_index()

        # Merge with actual data
        merged = panel_df.merge(
            df,
            on=['region', 'date', 'feature_name'],
            how='left'
        )

        return merged

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in panel.

        Args:
            df: Panel DataFrame

        Returns:
            DataFrame with missing data handled
        """
        strategy = config.get(
            'preprocessing.missing_data.strategy',
            'forward_fill'
        )

        max_gap = config.get(
            'preprocessing.missing_data.max_gap',
            7
        )

        logger.info(f"Handling missing data (strategy={strategy}, max_gap={max_gap})")

        # Group by region and feature
        grouped = df.groupby(['region', 'feature_name'])

        if strategy == 'forward_fill':
            df['value'] = grouped['value'].fillna(method='ffill', limit=max_gap)

        elif strategy == 'backward_fill':
            df['value'] = grouped['value'].fillna(method='bfill', limit=max_gap)

        elif strategy == 'interpolate':
            df['value'] = grouped['value'].apply(
                lambda x: x.interpolate(method='linear', limit=max_gap)
            )

        elif strategy == 'mean':
            df['value'] = grouped['value'].fillna(grouped['value'].transform('mean'))

        elif strategy == 'median':
            df['value'] = grouped['value'].fillna(grouped['value'].transform('median'))

        elif strategy == 'zero':
            df['value'] = df['value'].fillna(0)

        # Check remaining missing data
        missing_ratio = df['value'].isnull().sum() / len(df)

        if missing_ratio > self.max_missing_ratio:
            logger.warning(
                f"High missing data ratio after imputation: {missing_ratio:.2%}"
            )

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from harmonized data.

        Args:
            df: Harmonized panel DataFrame

        Returns:
            DataFrame with additional features
        """
        logger.info("Creating engineered features")

        # Pivot to wide format for feature engineering
        wide_df = df.pivot_table(
            index=['region', 'date'],
            columns='feature_name',
            values='value'
        ).reset_index()

        # Create lag features
        lag_config = config.get('preprocessing.features.lag_features', {})
        if lag_config.get('enabled', True):
            lags = lag_config.get('lags', [1, 3, 7, 14, 30])
            wide_df = self._create_lag_features(wide_df, lags)

        # Create rolling features
        rolling_config = config.get('preprocessing.features.rolling_features', {})
        if rolling_config.get('enabled', True):
            windows = rolling_config.get('windows', [7, 14, 30, 90])
            functions = rolling_config.get('functions', ['mean', 'std'])
            wide_df = self._create_rolling_features(wide_df, windows, functions)

        # Create cyclical time features
        cyclical_config = config.get('preprocessing.features.cyclical_features', {})
        if cyclical_config.get('enabled', True):
            wide_df = self._create_cyclical_features(wide_df)

        # Convert back to long format
        long_df = wide_df.melt(
            id_vars=['region', 'date'],
            var_name='feature_name',
            value_name='value'
        )

        long_df = long_df.dropna(subset=['value'])

        logger.info(
            f"Feature engineering complete: {long_df['feature_name'].nunique()} features"
        )

        return long_df

    def _create_lag_features(
        self,
        df: pd.DataFrame,
        lags: List[int]
    ) -> pd.DataFrame:
        """Create lagged features.

        Args:
            df: Wide-format DataFrame
            lags: List of lag periods

        Returns:
            DataFrame with lag features added
        """
        feature_cols = [col for col in df.columns if col not in ['region', 'date']]

        for col in feature_cols:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df.groupby('region')[col].shift(lag)

        return df

    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        windows: List[int],
        functions: List[str]
    ) -> pd.DataFrame:
        """Create rolling window features.

        Args:
            df: Wide-format DataFrame
            windows: List of window sizes
            functions: List of aggregation functions

        Returns:
            DataFrame with rolling features added
        """
        feature_cols = [col for col in df.columns if col not in ['region', 'date']]

        for col in feature_cols:
            for window in windows:
                grouped = df.groupby('region')[col].rolling(window, min_periods=1)

                for func in functions:
                    if func == 'mean':
                        df[f"{col}_roll_{window}_mean"] = grouped.mean().reset_index(0, drop=True)
                    elif func == 'std':
                        df[f"{col}_roll_{window}_std"] = grouped.std().reset_index(0, drop=True)
                    elif func == 'min':
                        df[f"{col}_roll_{window}_min"] = grouped.min().reset_index(0, drop=True)
                    elif func == 'max':
                        df[f"{col}_roll_{window}_max"] = grouped.max().reset_index(0, drop=True)

        return df

    def _create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical time features.

        Args:
            df: DataFrame with date column

        Returns:
            DataFrame with cyclical features added
        """
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter

        # Sine/cosine encoding for cyclical features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def save_harmonized(
        self,
        df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> Path:
        """Save harmonized data to disk.

        Args:
            df: Harmonized DataFrame
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"harmonized_{timestamp}.parquet"

        filepath = self.processed_data_dir / filename

        df.to_parquet(filepath, compression='snappy', index=False)

        logger.info(f"Saved harmonized data to {filepath}")

        return filepath

    def load_harmonized(self, filename: str) -> pd.DataFrame:
        """Load harmonized data from disk.

        Args:
            filename: Name of file to load

        Returns:
            Harmonized DataFrame
        """
        filepath = self.processed_data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_parquet(filepath)

        logger.info(f"Loaded harmonized data from {filepath}")

        return df
