"""Baseline forecasting models for risk indicator prediction.

Implements traditional time-series forecasting methods (Prophet and ARIMA)
to provide baseline comparisons for advanced models.
"""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

from pi_ere.utils.config import config


class BaselineForecaster:
    """ARIMA/Prophet baseline for risk indicator forecasting.

    Provides traditional statistical forecasting methods as baselines for
    comparing against deep learning approaches.
    """

    def __init__(self, model_type: str = 'prophet'):
        """Initialize baseline forecaster.

        Args:
            model_type: Type of model to use ('prophet' or 'arima')

        Raises:
            ValueError: If model_type is not 'prophet' or 'arima'
        """
        if model_type not in ['prophet', 'arima']:
            raise ValueError(f"model_type must be 'prophet' or 'arima', got '{model_type}'")

        self.model_type = model_type
        self.model: Optional[Union[Prophet, ARIMA]] = None
        self.is_fitted = False
        self._last_date: Optional[pd.Timestamp] = None
        self._freq: Optional[str] = None
        self._confidence: float = 0.95

        logger.info(f"Initialized BaselineForecaster (model_type={model_type})")

    def fit(
        self,
        series: pd.Series,
        date_column: Optional[str] = None
    ) -> 'BaselineForecaster':
        """Fit the forecasting model to time-series data.

        Args:
            series: Time-series data with datetime index or DatetimeIndex
            date_column: Optional column name if series is not indexed by date

        Returns:
            Self for method chaining

        Raises:
            ValueError: If series is invalid or has insufficient data
        """
        # Validate and prepare data
        df = self._prepare_data(series, date_column)

        logger.info(
            f"Fitting {self.model_type} model on {len(df)} observations "
            f"from {df['ds'].min()} to {df['ds'].max()}"
        )

        # Handle missing values
        if df['y'].isna().any():
            n_missing = df['y'].isna().sum()
            logger.warning(f"Found {n_missing} missing values, interpolating...")
            df['y'] = df['y'].interpolate(method='time', limit_direction='both')

            # Fill any remaining NaNs at edges
            df['y'] = df['y'].fillna(df['y'].mean())

        # Fit model
        if self.model_type == 'prophet':
            self._fit_prophet(df)
        else:
            self._fit_arima(df)

        self.is_fitted = True
        self._last_date = df['ds'].max()
        self._freq = pd.infer_freq(df['ds'])

        logger.info(f"Model fitted successfully (freq={self._freq})")

        return self

    def predict(self, horizon: int = 180) -> pd.DataFrame:
        """Generate forecasts for future time periods.

        Args:
            horizon: Number of time steps to forecast ahead (default: 180 days)

        Returns:
            DataFrame with columns: date, yhat, yhat_lower, yhat_upper

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        logger.info(f"Generating {horizon}-step forecast")

        if self.model_type == 'prophet':
            forecast = self._predict_prophet(horizon)
        else:
            forecast = self._predict_arima(horizon)

        return forecast

    def get_prediction_intervals(self, confidence: float = 0.95) -> pd.DataFrame:
        """Get prediction intervals with configurable confidence level.

        Note: For already-generated predictions, this method adjusts the intervals
        based on the new confidence level.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% intervals)

        Returns:
            DataFrame with adjusted prediction intervals

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting intervals")

        self._confidence = confidence

        logger.info(f"Updating prediction intervals (confidence={confidence})")

        # Re-run prediction with new confidence level
        horizon = config.get('forecasting.forecast_horizon', 90)
        return self.predict(horizon=horizon)

    def save(self, path: Path):
        """Save fitted model to disk.

        Args:
            path: Path to save the model (will create parent directories)
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare state for serialization
        state = {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'last_date': self._last_date,
            'freq': self._freq,
            'confidence': self._confidence,
        }

        # Save model separately based on type
        if self.model_type == 'prophet':
            # Prophet models have their own serialization
            with open(path, 'wb') as f:
                pickle.dump({'state': state, 'model': self.model}, f)
        else:
            # ARIMA models can be pickled directly
            state['model'] = self.model
            with open(path, 'wb') as f:
                pickle.dump(state, f)

        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path) -> 'BaselineForecaster':
        """Load a fitted model from disk.

        Args:
            path: Path to the saved model file

        Returns:
            Loaded BaselineForecaster instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Handle both serialization formats
        if 'state' in data:
            state = data['state']
            model = data['model']
        else:
            state = data
            model = data.get('model')

        # Create instance and restore state
        instance = cls(model_type=state['model_type'])
        instance.model = model
        instance.is_fitted = state['is_fitted']
        instance._last_date = state['last_date']
        instance._freq = state['freq']
        instance._confidence = state['confidence']

        logger.info(f"Loaded model from {path}")

        return instance

    def _prepare_data(
        self,
        series: pd.Series,
        date_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Prepare time-series data for modeling.

        Args:
            series: Input time-series
            date_column: Optional date column name

        Returns:
            DataFrame with 'ds' (date) and 'y' (value) columns
        """
        if len(series) < 10:
            raise ValueError(f"Insufficient data: need at least 10 observations, got {len(series)}")

        # Convert to DataFrame if needed
        if isinstance(series, pd.Series):
            df = pd.DataFrame({'y': series})

            if date_column:
                df['ds'] = series.index.get_level_values(date_column)
            else:
                if not isinstance(series.index, pd.DatetimeIndex):
                    raise ValueError("Series must have DatetimeIndex or date_column must be specified")
                df['ds'] = series.index
        else:
            raise TypeError(f"Expected pd.Series, got {type(series)}")

        # Ensure datetime type
        df['ds'] = pd.to_datetime(df['ds'])

        # Sort by date
        df = df.sort_values('ds').reset_index(drop=True)

        return df[['ds', 'y']]

    def _fit_prophet(self, df: pd.DataFrame):
        """Fit Prophet model with suppressed output.

        Args:
            df: DataFrame with 'ds' and 'y' columns
        """
        import logging

        # Suppress Prophet's verbose logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

        self.model = Prophet(
            interval_width=self._confidence,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
        )

        self.model.fit(df)

    def _fit_arima(self, df: pd.DataFrame):
        """Fit ARIMA model with automatic order selection.

        Args:
            df: DataFrame with 'ds' and 'y' columns
        """
        # Use auto ARIMA for order selection (simplified version)
        # Default to ARIMA(1,1,1) for simplicity
        order = (1, 1, 1)

        self.model = ARIMA(df['y'].values, order=order)
        self.model = self.model.fit()

    def _predict_prophet(self, horizon: int) -> pd.DataFrame:
        """Generate Prophet predictions.

        Args:
            horizon: Forecast horizon

        Returns:
            Forecast DataFrame
        """
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=horizon)

        # Generate forecast
        forecast = self.model.predict(future)

        # Return only future predictions
        forecast = forecast.tail(horizon)

        return pd.DataFrame({
            'date': forecast['ds'].values,
            'yhat': forecast['yhat'].values,
            'yhat_lower': forecast['yhat_lower'].values,
            'yhat_upper': forecast['yhat_upper'].values,
        })

    def _predict_arima(self, horizon: int) -> pd.DataFrame:
        """Generate ARIMA predictions.

        Args:
            horizon: Forecast horizon

        Returns:
            Forecast DataFrame
        """
        # Get predictions with confidence intervals
        forecast_result = self.model.get_forecast(steps=horizon)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=1 - self._confidence)

        # Generate future dates
        freq = self._freq or 'D'
        future_dates = pd.date_range(
            start=self._last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq=freq
        )

        return pd.DataFrame({
            'date': future_dates,
            'yhat': forecast,
            'yhat_lower': conf_int.iloc[:, 0].values,
            'yhat_upper': conf_int.iloc[:, 1].values,
        })
