"""World Bank API connector for macroeconomic and governance indicators.

Fetches economic and governance data from the World Bank's World Development Indicators
and Worldwide Governance Indicators databases.
"""

from datetime import datetime
from typing import List, Optional

import pandas as pd
import wbgapi as wb
from loguru import logger

from pi_ere.data.ingest import DataSource


class WorldBankSource(DataSource):
    """Connector for World Bank macroeconomic and governance data.

    Uses the wbgapi library to fetch data from World Bank databases.
    Documentation: https://github.com/tgherzog/wbgapi
    """

    def __init__(self):
        """Initialize World Bank data source."""
        super().__init__(name='world_bank')

        # Get indicators from config
        self.indicators = self.config.get('indicators', {})

        if isinstance(self.indicators, list):
            # Legacy format: just a list of codes
            self.indicator_codes = self.indicators
        else:
            # New format: dict with code and name
            self.indicator_codes = [
                ind['code'] if isinstance(ind, dict) else ind
                for ind in self.indicators.values()
            ]

        logger.info(
            f"Initialized World Bank source with {len(self.indicator_codes)} indicators"
        )

    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        regions: Optional[List[str]] = None,
        indicators: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch World Bank indicators for specified date range and regions.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            regions: List of ISO country codes (e.g., ['COD', 'MLI'])
            indicators: List of indicator codes (None = use configured indicators)
            **kwargs: Additional parameters

        Returns:
            DataFrame with World Bank indicator data
        """
        if not self.enabled:
            logger.warning("World Bank source is disabled")
            return pd.DataFrame()

        indicators = indicators or self.indicator_codes

        if not indicators:
            logger.error("No indicators specified for World Bank fetch")
            return pd.DataFrame()

        if regions is None:
            from pi_ere.data.ingest import get_default_regions
            regions = get_default_regions()

        # World Bank data is annual/quarterly, so we use years
        start_year = start_date.year
        end_year = end_date.year

        logger.info(
            f"Fetching World Bank data for {len(regions)} regions, "
            f"{len(indicators)} indicators, years {start_year}-{end_year}"
        )

        try:
            # Fetch data using wbgapi
            # wbgapi returns data in long format
            data = wb.data.DataFrame(
                indicators,
                regions,
                time=range(start_year, end_year + 1),
                labels=True,
                numericTimeKeys=True,
                timeColumns=True,
            )

            if data.empty:
                logger.warning("No World Bank data fetched")
                return pd.DataFrame()

            # Convert to long format
            df = self._pivot_to_long_format(data, indicators)

            # Standardize columns
            df = self._standardize_columns(df)

            logger.info(f"Fetched {len(df)} World Bank data points")

            return df

        except Exception as e:
            logger.error(f"Error fetching World Bank data: {e}")
            logger.exception(e)
            return pd.DataFrame()

    def _pivot_to_long_format(
        self,
        data: pd.DataFrame,
        indicators: List[str]
    ) -> pd.DataFrame:
        """Convert wide-format World Bank data to long format.

        Args:
            data: Wide-format DataFrame from wbgapi
            indicators: List of indicator codes

        Returns:
            Long-format DataFrame
        """
        # Reset index to get economy (country) as column
        df = data.reset_index()

        # Melt year columns to long format
        year_columns = [col for col in df.columns if isinstance(col, int) or col.isdigit()]

        df_long = df.melt(
            id_vars=['economy'],
            value_vars=year_columns,
            var_name='year',
            value_name='value'
        )

        # If multiple indicators, they'll be in the index or columns
        # For now, we assume we're fetching one indicator at a time or they're separate rows
        return df_long

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize World Bank columns to PI-ERE schema.

        Args:
            df: Raw World Bank DataFrame

        Returns:
            Standardized DataFrame
        """
        # Rename columns
        column_mapping = {
            'economy': 'region',
            'year': 'year',
            'value': 'value',
        }

        df = df.rename(columns=column_mapping)

        # Convert year to datetime (using January 1st of each year)
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['date'] = pd.to_datetime(df['year'], format='%Y')

        # Add source column
        df['source'] = 'world_bank'

        # Convert value to numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Remove rows with null values
        df = df.dropna(subset=['value'])

        # Sort by region and date
        if 'date' in df.columns:
            df = df.sort_values(['region', 'date'])

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate World Bank data quality.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes
        """
        if df.empty:
            logger.warning("World Bank: DataFrame is empty")
            return True

        # Required columns
        required_columns = ['region', 'source']
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            logger.error(f"World Bank: Missing required columns: {missing_columns}")
            return False

        # Check for null values
        if 'value' in df.columns:
            null_count = df['value'].isnull().sum()
            if null_count > 0:
                logger.warning(f"World Bank: {null_count} null values")

        # Check value ranges (basic sanity checks)
        if 'value' in df.columns:
            inf_count = df['value'].isin([float('inf'), float('-inf')]).sum()
            if inf_count > 0:
                logger.warning(f"World Bank: {inf_count} infinite values")
                df = df[~df['value'].isin([float('inf'), float('-inf')])]

        logger.info("World Bank: Validation passed")
        return True

    def get_indicator_metadata(self, indicator_code: str) -> dict:
        """Get metadata for a specific indicator.

        Args:
            indicator_code: World Bank indicator code

        Returns:
            Dictionary with indicator metadata
        """
        try:
            # Fetch indicator metadata
            indicator = wb.series.get(indicator_code)

            metadata = {
                'code': indicator_code,
                'name': indicator.get('value', ''),
                'source': indicator.get('source', {}).get('value', ''),
                'description': indicator.get('sourceNote', ''),
                'topics': indicator.get('topics', []),
            }

            return metadata

        except Exception as e:
            logger.error(f"Error fetching metadata for {indicator_code}: {e}")
            return {'code': indicator_code}

    def fetch_governance_indicators(
        self,
        start_date: datetime,
        end_date: datetime,
        regions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fetch Worldwide Governance Indicators (WGI).

        The WGI includes:
        - Voice and Accountability
        - Political Stability and Absence of Violence
        - Government Effectiveness
        - Regulatory Quality
        - Rule of Law
        - Control of Corruption

        Args:
            start_date: Start date
            end_date: End date
            regions: List of ISO country codes

        Returns:
            DataFrame with governance indicators
        """
        governance_indicators = [
            'CC.EST',  # Control of Corruption
            'GE.EST',  # Government Effectiveness
            'PV.EST',  # Political Stability and Absence of Violence
            'RQ.EST',  # Regulatory Quality
            'RL.EST',  # Rule of Law
            'VA.EST',  # Voice and Accountability
        ]

        return self.fetch(
            start_date=start_date,
            end_date=end_date,
            regions=regions,
            indicators=governance_indicators,
        )
