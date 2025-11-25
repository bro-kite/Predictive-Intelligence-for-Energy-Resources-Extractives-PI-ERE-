"""GDELT (Global Database of Events, Language, and Tone) connector.

GDELT provides global news event data with sentiment and tone analysis.
This connector fetches event data for specified regions and time periods.
"""

import io
import zipfile
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import requests
from loguru import logger

from pi_ere.data.ingest import DataSource


class GDELTSource(DataSource):
    """Connector for GDELT event data.

    GDELT provides open data without API keys. Data is available as CSV files
    organized by day. Documentation: https://www.gdeltproject.org/
    """

    def __init__(self):
        """Initialize GDELT data source."""
        super().__init__(name='gdelt')

        self.base_url = self.config.get(
            'base_url',
            'http://data.gdeltproject.org/events'
        )

        # CAMEO event codes for relevant events
        # https://www.gdeltproject.org/data/documentation/CAMEO.Manual.1.1b3.pdf
        self.relevant_event_codes = self.config.get('event_codes', [
            '14',  # Protest
            '145', # Protest demonstrations
            '1451', # Demonstrate for leadership change
            '18',  # Assault
            '180', # Use unconventional violence
            '19',  # Fight
            '190', # Use conventional military force
            '20',  # Use unconventional mass violence
        ])

        # Tone threshold (negative = conflict/negative events)
        self.tone_threshold = self.config.get('tone_threshold', -5.0)

    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        regions: Optional[List[str]] = None,
        event_codes: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch GDELT events for specified date range.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            regions: List of ISO country codes (optional, for filtering)
            event_codes: List of CAMEO event codes to filter (None = use defaults)
            **kwargs: Additional parameters

        Returns:
            DataFrame with GDELT events
        """
        if not self.enabled:
            logger.warning("GDELT source is disabled")
            return pd.DataFrame()

        event_codes = event_codes or self.relevant_event_codes

        logger.info(
            f"Fetching GDELT data from {start_date.date()} to {end_date.date()}"
        )

        all_data = []
        current_date = start_date

        while current_date <= end_date:
            try:
                df = self._fetch_day(current_date)

                if not df.empty:
                    # Filter by event codes
                    if event_codes:
                        df = df[df['EventRootCode'].isin(event_codes)]

                    # Filter by regions if specified
                    if regions:
                        df = df[
                            df['ActionGeo_CountryCode'].isin(regions) |
                            df['Actor1Geo_CountryCode'].isin(regions) |
                            df['Actor2Geo_CountryCode'].isin(regions)
                        ]

                    # Filter by tone (negative events)
                    if self.tone_threshold is not None:
                        df = df[df['AvgTone'] <= self.tone_threshold]

                    if not df.empty:
                        all_data.append(df)
                        logger.debug(
                            f"Fetched {len(df)} GDELT events for {current_date.date()}"
                        )

            except Exception as e:
                logger.error(f"Error fetching GDELT data for {current_date.date()}: {e}")

            current_date += timedelta(days=1)

        if not all_data:
            logger.warning("No GDELT data fetched")
            return pd.DataFrame()

        # Combine all days
        combined_df = pd.concat(all_data, ignore_index=True)

        # Standardize columns
        combined_df = self._standardize_columns(combined_df)

        logger.info(f"Fetched {len(combined_df)} GDELT events")

        return combined_df

    def _fetch_day(self, date: datetime) -> pd.DataFrame:
        """Fetch GDELT data for a single day.

        Args:
            date: Date to fetch

        Returns:
            DataFrame with events for this day
        """
        # GDELT file naming: YYYYMMDD.export.CSV.zip
        date_str = date.strftime('%Y%m%d')
        filename = f"{date_str}.export.CSV.zip"
        url = f"{self.base_url}/{filename}"

        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Unzip and read CSV
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_filename = f"{date_str}.export.CSV"

                with zf.open(csv_filename) as csv_file:
                    # GDELT doesn't have headers, so we need to add them
                    df = pd.read_csv(
                        csv_file,
                        sep='\t',
                        header=None,
                        names=self._get_gdelt_columns(),
                        encoding='utf-8',
                        low_memory=False,
                        on_bad_lines='skip',
                    )

            return df

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"No GDELT data available for {date.date()}")
            else:
                logger.error(f"HTTP error fetching GDELT data for {date.date()}: {e}")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching GDELT data for {date.date()}: {e}")
            return pd.DataFrame()

    def _get_gdelt_columns(self) -> List[str]:
        """Get GDELT column names.

        Returns:
            List of column names for GDELT 1.0 format
        """
        # GDELT 1.0 Event Database column definitions
        return [
            'GlobalEventID',
            'Day',
            'MonthYear',
            'Year',
            'FractionDate',
            'Actor1Code',
            'Actor1Name',
            'Actor1CountryCode',
            'Actor1KnownGroupCode',
            'Actor1EthnicCode',
            'Actor1Religion1Code',
            'Actor1Religion2Code',
            'Actor1Type1Code',
            'Actor1Type2Code',
            'Actor1Type3Code',
            'Actor2Code',
            'Actor2Name',
            'Actor2CountryCode',
            'Actor2KnownGroupCode',
            'Actor2EthnicCode',
            'Actor2Religion1Code',
            'Actor2Religion2Code',
            'Actor2Type1Code',
            'Actor2Type2Code',
            'Actor2Type3Code',
            'IsRootEvent',
            'EventCode',
            'EventBaseCode',
            'EventRootCode',
            'QuadClass',
            'GoldsteinScale',
            'NumMentions',
            'NumSources',
            'NumArticles',
            'AvgTone',
            'Actor1Geo_Type',
            'Actor1Geo_FullName',
            'Actor1Geo_CountryCode',
            'Actor1Geo_ADM1Code',
            'Actor1Geo_Lat',
            'Actor1Geo_Long',
            'Actor1Geo_FeatureID',
            'Actor2Geo_Type',
            'Actor2Geo_FullName',
            'Actor2Geo_CountryCode',
            'Actor2Geo_ADM1Code',
            'Actor2Geo_Lat',
            'Actor2Geo_Long',
            'Actor2Geo_FeatureID',
            'ActionGeo_Type',
            'ActionGeo_FullName',
            'ActionGeo_CountryCode',
            'ActionGeo_ADM1Code',
            'ActionGeo_Lat',
            'ActionGeo_Long',
            'ActionGeo_FeatureID',
            'DATEADDED',
        ]

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize GDELT columns to PI-ERE schema.

        Args:
            df: Raw GDELT DataFrame

        Returns:
            Standardized DataFrame
        """
        # Convert date (YYYYMMDD format) to datetime
        if 'Day' in df.columns:
            df['date'] = pd.to_datetime(df['Day'], format='%Y%m%d', errors='coerce')

        # Use action geography as primary location
        df['region'] = df.get('ActionGeo_CountryCode', '')
        df['country_name'] = df.get('ActionGeo_FullName', '')
        df['latitude'] = pd.to_numeric(df.get('ActionGeo_Lat', None), errors='coerce')
        df['longitude'] = pd.to_numeric(df.get('ActionGeo_Long', None), errors='coerce')

        # Event information
        df['event_code'] = df.get('EventCode', '')
        df['event_root_code'] = df.get('EventRootCode', '')
        df['goldstein_scale'] = pd.to_numeric(
            df.get('GoldsteinScale', None),
            errors='coerce'
        )
        df['tone'] = pd.to_numeric(df.get('AvgTone', None), errors='coerce')

        # Actors
        df['actor_1'] = df.get('Actor1Name', '')
        df['actor_1_country'] = df.get('Actor1CountryCode', '')
        df['actor_2'] = df.get('Actor2Name', '')
        df['actor_2_country'] = df.get('Actor2CountryCode', '')

        # Mentions and coverage
        df['num_mentions'] = pd.to_numeric(df.get('NumMentions', 0), errors='coerce')
        df['num_sources'] = pd.to_numeric(df.get('NumSources', 0), errors='coerce')
        df['num_articles'] = pd.to_numeric(df.get('NumArticles', 0), errors='coerce')

        # Add source column
        df['source'] = 'gdelt'

        # Select relevant columns
        columns_to_keep = [
            'date', 'region', 'country_name', 'latitude', 'longitude',
            'event_code', 'event_root_code', 'goldstein_scale', 'tone',
            'actor_1', 'actor_1_country', 'actor_2', 'actor_2_country',
            'num_mentions', 'num_sources', 'num_articles', 'source'
        ]

        df = df[[col for col in columns_to_keep if col in df.columns]]

        # Sort by date
        if 'date' in df.columns:
            df = df.sort_values('date')

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate GDELT data quality.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes
        """
        if df.empty:
            logger.warning("GDELT: DataFrame is empty")
            return True

        # Required columns
        required_columns = ['date', 'source']
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            logger.error(f"GDELT: Missing required columns: {missing_columns}")
            return False

        # Check for null dates
        if df['date'].isnull().any():
            null_count = df['date'].isnull().sum()
            logger.warning(f"GDELT: {null_count} records with null dates")
            df.dropna(subset=['date'], inplace=True)

        # Check date range
        if not df.empty:
            min_date = df['date'].min()
            max_date = df['date'].max()
            logger.info(f"GDELT: Date range: {min_date.date()} to {max_date.date()}")

        logger.info("GDELT: Validation passed")
        return True

    def aggregate_by_region_date(
        self,
        df: pd.DataFrame,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """Aggregate GDELT events by region and date.

        Args:
            df: GDELT DataFrame
            freq: Frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)

        Returns:
            Aggregated DataFrame with metrics by region/date
        """
        if df.empty:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df['period'] = df['date'].dt.to_period(freq)

        agg_dict = {
            'event_code': 'count',
            'goldstein_scale': 'mean',
            'tone': 'mean',
            'num_mentions': 'sum',
            'num_sources': 'sum',
            'num_articles': 'sum',
        }

        grouped = df.groupby(['region', 'period']).agg(agg_dict).reset_index()
        grouped.columns = [
            'region', 'period', 'event_count', 'avg_goldstein',
            'avg_tone', 'total_mentions', 'total_sources', 'total_articles'
        ]

        grouped['date'] = grouped['period'].dt.to_timestamp()
        grouped = grouped.drop('period', axis=1)

        return grouped
