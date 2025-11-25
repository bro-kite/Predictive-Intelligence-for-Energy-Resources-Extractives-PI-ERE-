"""ACLED (Armed Conflict Location & Event Data Project) connector.

ACLED provides real-time data on political violence and protest events worldwide.
This connector fetches conflict and protest data for specified regions and time periods.
"""

import os
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd
import requests
from loguru import logger

from pi_ere.data.ingest import DataSource


class ACLEDSource(DataSource):
    """Connector for ACLED conflict and protest data.

    API Documentation: https://acleddata.com/resources/general-guides/
    """

    def __init__(self):
        """Initialize ACLED data source."""
        super().__init__(name='acled')

        self.api_key = os.getenv(self.config.get('api_key_env', 'ACLED_API_KEY'))
        self.email = os.getenv('ACLED_EMAIL', '')
        self.base_url = self.config.get(
            'base_url',
            'https://api.acleddata.com/acled/read'
        )

        if not self.api_key:
            logger.warning(
                "ACLED API key not found. Set ACLED_API_KEY environment variable."
            )

        # Event type mapping
        self.event_types = self.config.get('event_types', [
            'Battles',
            'Explosions/Remote violence',
            'Violence against civilians',
            'Protests',
            'Riots',
            'Strategic developments',
        ])

    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        regions: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch ACLED events for specified date range and regions.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            regions: List of ISO country codes (e.g., ['COD', 'MLI'])
            event_types: List of event types to fetch (None = all configured types)
            **kwargs: Additional API parameters

        Returns:
            DataFrame with ACLED events
        """
        if not self.enabled:
            logger.warning("ACLED source is disabled")
            return pd.DataFrame()

        if not self.api_key:
            logger.error("Cannot fetch ACLED data without API key")
            return pd.DataFrame()

        event_types = event_types or self.event_types
        all_data = []

        # ACLED API has pagination limits, so we fetch by country
        if regions is None:
            from pi_ere.data.ingest import get_default_regions
            regions = get_default_regions()

        logger.info(
            f"Fetching ACLED data for {len(regions)} regions from "
            f"{start_date.date()} to {end_date.date()}"
        )

        for region in regions:
            try:
                df = self._fetch_region(
                    region=region,
                    start_date=start_date,
                    end_date=end_date,
                    event_types=event_types,
                )
                if not df.empty:
                    all_data.append(df)

                # Rate limiting: ACLED allows 60 requests/min
                time.sleep(1.1)

            except Exception as e:
                logger.error(f"Error fetching ACLED data for {region}: {e}")
                continue

        if not all_data:
            logger.warning("No ACLED data fetched")
            return pd.DataFrame()

        # Combine all regions
        combined_df = pd.concat(all_data, ignore_index=True)

        # Standardize columns
        combined_df = self._standardize_columns(combined_df)

        logger.info(f"Fetched {len(combined_df)} ACLED events")

        return combined_df

    def _fetch_region(
        self,
        region: str,
        start_date: datetime,
        end_date: datetime,
        event_types: List[str],
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Fetch ACLED data for a single region.

        Args:
            region: ISO country code
            start_date: Start date
            end_date: End date
            event_types: Event types to fetch
            limit: Maximum records per request

        Returns:
            DataFrame with events for this region
        """
        params = {
            'key': self.api_key,
            'email': self.email,
            'iso': region,
            'event_date': f"{start_date.strftime('%Y-%m-%d')}|{end_date.strftime('%Y-%m-%d')}",
            'event_date_where': 'BETWEEN',
            'limit': limit,
        }

        # Add event type filter if specified
        if event_types:
            params['event_type'] = '|'.join(event_types)

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'data' not in data:
                logger.warning(f"No data returned for {region}")
                return pd.DataFrame()

            df = pd.DataFrame(data['data'])

            logger.debug(f"Fetched {len(df)} events for {region}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {region}: {e}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Unexpected error for {region}: {e}")
            return pd.DataFrame()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize ACLED columns to PI-ERE schema.

        Args:
            df: Raw ACLED DataFrame

        Returns:
            Standardized DataFrame
        """
        # Rename columns to standard names
        column_mapping = {
            'event_date': 'date',
            'iso': 'region',
            'country': 'country_name',
            'event_type': 'event_type',
            'sub_event_type': 'sub_event_type',
            'actor1': 'actor_1',
            'actor2': 'actor_2',
            'fatalities': 'fatalities',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'notes': 'description',
            'disorder_type': 'disorder_type',
        }

        # Select and rename columns that exist
        existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_columns)

        # Convert date to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Add source column
        df['source'] = 'acled'

        # Convert fatalities to numeric
        if 'fatalities' in df.columns:
            df['fatalities'] = pd.to_numeric(df['fatalities'], errors='coerce').fillna(0)

        # Convert coordinates to numeric
        for col in ['latitude', 'longitude']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date')

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate ACLED data quality.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes
        """
        if df.empty:
            logger.warning("ACLED: DataFrame is empty")
            return True  # Empty is technically valid

        # Required columns
        required_columns = ['date', 'region', 'source']
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            logger.error(f"ACLED: Missing required columns: {missing_columns}")
            return False

        # Check for null dates
        if df['date'].isnull().any():
            null_count = df['date'].isnull().sum()
            logger.warning(f"ACLED: {null_count} records with null dates")

            # Remove rows with null dates
            df.dropna(subset=['date'], inplace=True)

        # Check date range validity
        if not df.empty:
            min_date = df['date'].min()
            max_date = df['date'].max()
            logger.info(f"ACLED: Date range: {min_date.date()} to {max_date.date()}")

        # Check for valid coordinates
        if 'latitude' in df.columns and 'longitude' in df.columns:
            invalid_coords = (
                (df['latitude'].abs() > 90) |
                (df['longitude'].abs() > 180)
            ).sum()

            if invalid_coords > 0:
                logger.warning(f"ACLED: {invalid_coords} records with invalid coordinates")

        # Check missing data ratio
        missing_ratio = df.isnull().sum() / len(df)
        high_missing = missing_ratio[missing_ratio > 0.5]

        if not high_missing.empty:
            logger.warning(
                f"ACLED: High missing data ratio for columns: {high_missing.to_dict()}"
            )

        logger.info("ACLED: Validation passed")
        return True

    def aggregate_by_region_date(
        self,
        df: pd.DataFrame,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """Aggregate ACLED events by region and date.

        Args:
            df: ACLED DataFrame
            freq: Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)

        Returns:
            Aggregated DataFrame with counts and metrics by region/date
        """
        if df.empty:
            return pd.DataFrame()

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Group by region and date period
        df['period'] = df['date'].dt.to_period(freq)

        agg_dict = {
            'event_type': 'count',  # Total events
            'fatalities': 'sum',
        }

        # Add event type counts
        if 'event_type' in df.columns:
            for event_type in df['event_type'].unique():
                df[f'is_{event_type.lower().replace(" ", "_")}'] = (
                    df['event_type'] == event_type
                ).astype(int)

        grouped = df.groupby(['region', 'period']).agg(agg_dict).reset_index()
        grouped.columns = ['region', 'period', 'event_count', 'total_fatalities']

        # Convert period back to timestamp
        grouped['date'] = grouped['period'].dt.to_timestamp()
        grouped = grouped.drop('period', axis=1)

        return grouped
