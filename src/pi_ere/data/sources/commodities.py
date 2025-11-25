"""Commodities price data connector.

Fetches commodity prices from multiple sources:
- FRED (Federal Reserve Economic Data)
- Yahoo Finance
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from pi_ere.data.ingest import DataSource


class CommoditiesSource(DataSource):
    """Connector for commodity price data.

    Fetches prices for key commodities relevant to extractive industries:
    - Oil (WTI, Brent)
    - Gold
    - Copper
    - Natural Gas
    - Coal
    """

    def __init__(self):
        """Initialize Commodities data source."""
        super().__init__(name='commodities')

        self.sources_config = self.config.get('sources', {})
        self.fred_config = self.sources_config.get('fred', {})
        self.yahoo_config = self.sources_config.get('yahoo_finance', {})

        self.fred_api_key = os.getenv('FRED_API_KEY', '')

        if not self.fred_api_key:
            logger.warning(
                "FRED API key not found. Some commodity data may be unavailable. "
                "Set FRED_API_KEY environment variable."
            )

    def fetch(
        self,
        start_date: datetime,
        end_date: datetime,
        regions: Optional[List[str]] = None,
        commodities: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch commodity prices for specified date range.

        Args:
            start_date: Start date for data fetch
            end_date: End date for data fetch
            regions: Not used for commodities (global prices)
            commodities: List of commodity names (None = all configured)
            **kwargs: Additional parameters

        Returns:
            DataFrame with commodity price data
        """
        if not self.enabled:
            logger.warning("Commodities source is disabled")
            return pd.DataFrame()

        logger.info(
            f"Fetching commodity prices from {start_date.date()} to {end_date.date()}"
        )

        all_data = []

        # Fetch from FRED
        if self.fred_config.get('enabled', True) and self.fred_api_key:
            fred_data = self._fetch_from_fred(start_date, end_date)
            if not fred_data.empty:
                all_data.append(fred_data)

        # Fetch from Yahoo Finance
        if self.yahoo_config.get('enabled', True):
            yahoo_data = self._fetch_from_yahoo(start_date, end_date)
            if not yahoo_data.empty:
                all_data.append(yahoo_data)

        if not all_data:
            logger.warning("No commodity data fetched")
            return pd.DataFrame()

        # Combine all sources
        combined_df = pd.concat(all_data, ignore_index=True)

        # Standardize columns
        combined_df = self._standardize_columns(combined_df)

        logger.info(f"Fetched {len(combined_df)} commodity price records")

        return combined_df

    def _fetch_from_fred(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch commodity prices from FRED.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with FRED commodity prices
        """
        base_url = self.fred_config.get(
            'base_url',
            'https://api.stlouisfed.org/fred/series/observations'
        )

        series_config = self.fred_config.get('series', {})

        all_series_data = []

        for commodity_key, series_info in series_config.items():
            if isinstance(series_info, dict):
                series_id = series_info.get('id')
                commodity_name = series_info.get('name', commodity_key)
            else:
                # Legacy format: just the series ID
                series_id = series_info
                commodity_name = commodity_key

            try:
                df = self._fetch_fred_series(
                    series_id=series_id,
                    start_date=start_date,
                    end_date=end_date,
                    base_url=base_url,
                )

                if not df.empty:
                    df['commodity'] = commodity_name
                    df['commodity_code'] = series_id
                    all_series_data.append(df)

                logger.debug(f"Fetched {len(df)} records for {commodity_name} from FRED")

            except Exception as e:
                logger.error(f"Error fetching {series_id} from FRED: {e}")
                continue

        if not all_series_data:
            return pd.DataFrame()

        return pd.concat(all_series_data, ignore_index=True)

    def _fetch_fred_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime,
        base_url: str,
    ) -> pd.DataFrame:
        """Fetch a single FRED time series.

        Args:
            series_id: FRED series ID
            start_date: Start date
            end_date: End date
            base_url: FRED API base URL

        Returns:
            DataFrame with series data
        """
        params = {
            'series_id': series_id,
            'api_key': self.fred_api_key,
            'file_type': 'json',
            'observation_start': start_date.strftime('%Y-%m-%d'),
            'observation_end': end_date.strftime('%Y-%m-%d'),
        }

        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'observations' not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data['observations'])

        # Convert date and value
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Remove missing values (FRED uses '.' for missing)
        df = df.dropna(subset=['value'])

        df['data_source'] = 'fred'

        return df[['date', 'value', 'data_source']]

    def _fetch_from_yahoo(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch commodity prices from Yahoo Finance.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with Yahoo Finance commodity prices
        """
        tickers = self.yahoo_config.get('tickers', [])

        # Ticker to commodity name mapping
        ticker_names = {
            'GC=F': 'Gold Futures',
            'CL=F': 'Crude Oil Futures (WTI)',
            'BZ=F': 'Crude Oil Futures (Brent)',
            'HG=F': 'Copper Futures',
            'NG=F': 'Natural Gas Futures',
            'SI=F': 'Silver Futures',
        }

        all_data = []

        for ticker in tickers:
            try:
                # Use pandas_datareader or yfinance
                try:
                    import yfinance as yf
                    df = self._fetch_yahoo_yfinance(ticker, start_date, end_date)
                except ImportError:
                    logger.warning(
                        "yfinance not installed. Install with: pip install yfinance"
                    )
                    continue

                if not df.empty:
                    df['commodity'] = ticker_names.get(ticker, ticker)
                    df['commodity_code'] = ticker
                    all_data.append(df)

                logger.debug(
                    f"Fetched {len(df)} records for {ticker} from Yahoo Finance"
                )

            except Exception as e:
                logger.error(f"Error fetching {ticker} from Yahoo Finance: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def _fetch_yahoo_yfinance(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Fetch data using yfinance library.

        Args:
            ticker: Yahoo Finance ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with price data
        """
        import yfinance as yf

        data = yf.download(
            ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
        )

        if data.empty:
            return pd.DataFrame()

        # Use adjusted close price
        df = pd.DataFrame({
            'date': data.index,
            'value': data['Adj Close'].values if 'Adj Close' in data else data['Close'].values,
            'data_source': 'yahoo_finance',
        })

        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['value'])

        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize commodities columns to PI-ERE schema.

        Args:
            df: Raw commodities DataFrame

        Returns:
            Standardized DataFrame
        """
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Add source column
        df['source'] = 'commodities'

        # Region is global for commodities
        df['region'] = 'GLOBAL'

        # Convert value to numeric
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Rename for clarity
        if 'value' in df.columns:
            df = df.rename(columns={'value': 'price'})

        # Sort by commodity and date
        if 'date' in df.columns and 'commodity' in df.columns:
            df = df.sort_values(['commodity', 'date'])

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate commodities data quality.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes
        """
        if df.empty:
            logger.warning("Commodities: DataFrame is empty")
            return True

        # Required columns
        required_columns = ['date', 'source']
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            logger.error(f"Commodities: Missing required columns: {missing_columns}")
            return False

        # Check for null dates
        if df['date'].isnull().any():
            null_count = df['date'].isnull().sum()
            logger.warning(f"Commodities: {null_count} records with null dates")
            df.dropna(subset=['date'], inplace=True)

        # Check for negative prices (shouldn't happen for commodities)
        if 'price' in df.columns:
            negative_prices = (df['price'] < 0).sum()
            if negative_prices > 0:
                logger.warning(f"Commodities: {negative_prices} negative prices found")

        # Check date range
        if not df.empty:
            min_date = df['date'].min()
            max_date = df['date'].max()
            logger.info(f"Commodities: Date range: {min_date.date()} to {max_date.date()}")

        logger.info("Commodities: Validation passed")
        return True

    def get_commodity_returns(
        self,
        df: pd.DataFrame,
        window: int = 1
    ) -> pd.DataFrame:
        """Calculate commodity price returns.

        Args:
            df: Commodities DataFrame with price column
            window: Window for return calculation (days)

        Returns:
            DataFrame with returns added
        """
        if 'price' not in df.columns or 'commodity' not in df.columns:
            logger.error("Cannot calculate returns: missing price or commodity columns")
            return df

        # Sort by commodity and date
        df = df.sort_values(['commodity', 'date'])

        # Calculate returns for each commodity
        df['return'] = df.groupby('commodity')['price'].pct_change(periods=window)

        # Calculate log returns
        df['log_return'] = df.groupby('commodity')['price'].apply(
            lambda x: (x / x.shift(window)).apply(lambda y: 0 if y <= 0 else pd.np.log(y))
        )

        return df
