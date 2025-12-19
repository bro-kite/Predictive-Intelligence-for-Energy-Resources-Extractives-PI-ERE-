"""Risk forecasting module combining embeddings and baseline forecasts.

Provides a unified interface for generating risk forecasts by combining
time-series embeddings, text embeddings, and baseline statistical forecasts.
"""

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from pi_ere.models.baseline import BaselineForecaster


@dataclass
class RiskForecast:
    """Risk forecast output containing predictions and metadata.

    Attributes:
        region: Region identifier
        forecast_date: Date when forecast was generated
        horizon_days: Forecast horizon in days
        risk_scores: DataFrame with columns: date, score, lower, upper
        drivers: List of top feature names driving the forecast
        confidence: Overall forecast confidence (0-1)
        metadata: Optional additional metadata
    """
    region: str
    forecast_date: datetime
    horizon_days: int
    risk_scores: pd.DataFrame
    drivers: List[str]
    confidence: float
    metadata: Dict = field(default_factory=dict)


class RiskForecaster:
    """Main forecasting interface for operational risk.

    Combines time-series embeddings, text embeddings, and baseline forecasts
    to generate comprehensive risk predictions with feature attribution.
    """

    def __init__(
        self,
        ts_embedder=None,
        text_embedder=None,
        baseline_type: str = 'prophet'
    ):
        """Initialize risk forecaster.

        Args:
            ts_embedder: Optional TimeSeriesEmbedder instance
            text_embedder: Optional TextEmbedder instance
            baseline_type: Type of baseline model ('prophet' or 'arima')
        """
        self.ts_embedder = ts_embedder
        self.text_embedder = text_embedder
        self.baseline_type = baseline_type

        # State variables
        self.baseline_model: Optional[BaselineForecaster] = None
        self.fitted_region: Optional[str] = None
        self.feature_names: List[str] = []
        self.feature_weights: Dict[str, float] = {}
        self.historical_data: Optional[pd.DataFrame] = None

        logger.info(
            f"Initialized RiskForecaster "
            f"(baseline_type={baseline_type}, "
            f"has_ts_embedder={ts_embedder is not None}, "
            f"has_text_embedder={text_embedder is not None})"
        )

    def fit(
        self,
        harmonized_data: pd.DataFrame,
        region: str,
        target_col: str = 'risk_score'
    ) -> 'RiskForecaster':
        """Fit forecaster on harmonized data for a specific region.

        Args:
            harmonized_data: Long-format DataFrame with columns:
                            region, date, feature_name, value
            region: Region identifier to fit model for
            target_col: Name of target column (default: 'risk_score')

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is invalid or region not found
        """
        logger.info(f"Fitting RiskForecaster for region '{region}'")

        # Validate data format
        required_cols = ['region', 'date', 'feature_name', 'value']
        missing_cols = [col for col in required_cols if col not in harmonized_data.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Filter to specified region
        region_data = harmonized_data[harmonized_data['region'] == region].copy()

        if len(region_data) == 0:
            raise ValueError(f"No data found for region '{region}'")

        logger.info(f"Found {len(region_data)} records for region '{region}'")

        # Convert to wide format (pivot)
        wide_data = self._pivot_to_wide(region_data)

        # Store for later use
        self.historical_data = wide_data
        self.feature_names = [col for col in wide_data.columns if col != target_col]
        self.fitted_region = region

        # Create or validate risk score
        if target_col not in wide_data.columns:
            logger.info(f"Target column '{target_col}' not found, creating synthetic risk score")
            wide_data[target_col] = self._create_synthetic_risk_score(wide_data)

        # Fit baseline forecaster
        risk_series = wide_data[target_col]

        self.baseline_model = BaselineForecaster(model_type=self.baseline_type)
        self.baseline_model.fit(risk_series)

        # Compute feature importance/weights
        self._compute_feature_weights(wide_data, target_col)

        logger.info(
            f"Model fitted successfully "
            f"({len(self.feature_names)} features, "
            f"{len(wide_data)} time steps)"
        )

        return self

    def predict_risk(
        self,
        region: str,
        horizon_months: int = 6
    ) -> RiskForecast:
        """Generate risk forecast for specified region.

        Args:
            region: Region identifier
            horizon_months: Forecast horizon in months

        Returns:
            RiskForecast object containing predictions and metadata

        Raises:
            RuntimeError: If model not fitted or region mismatch
        """
        if self.baseline_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if region != self.fitted_region:
            raise ValueError(
                f"Model fitted for region '{self.fitted_region}', "
                f"cannot predict for '{region}'"
            )

        # Convert months to days (approximate)
        horizon_days = horizon_months * 30

        logger.info(
            f"Generating {horizon_months}-month ({horizon_days}-day) "
            f"risk forecast for region '{region}'"
        )

        # Generate baseline forecast
        forecast_df = self.baseline_model.predict(horizon=horizon_days)

        # Rename columns to match expected format
        risk_scores = pd.DataFrame({
            'date': forecast_df['date'],
            'score': forecast_df['yhat'],
            'lower': forecast_df['yhat_lower'],
            'upper': forecast_df['yhat_upper'],
        })

        # Identify top drivers
        drivers = self._identify_drivers()

        # Compute confidence based on forecast uncertainty
        confidence = self._compute_confidence(risk_scores)

        # Create forecast object
        forecast = RiskForecast(
            region=region,
            forecast_date=datetime.now(),
            horizon_days=horizon_days,
            risk_scores=risk_scores,
            drivers=drivers,
            confidence=confidence,
            metadata={
                'baseline_type': self.baseline_type,
                'n_features': len(self.feature_names),
                'n_historical_points': len(self.historical_data) if self.historical_data is not None else 0,
            }
        )

        logger.info(
            f"Forecast generated: {len(risk_scores)} predictions, "
            f"confidence={confidence:.3f}, "
            f"top_driver={drivers[0] if drivers else 'None'}"
        )

        return forecast

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores

        Raises:
            RuntimeError: If model not fitted
        """
        if not self.feature_weights:
            raise RuntimeError("Model not fitted. Call fit() first.")

        return self.feature_weights.copy()

    def save(self, path: Path):
        """Save forecaster state to disk.

        Args:
            path: Path to save forecaster
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save baseline model separately
        baseline_path = path.parent / f"{path.stem}_baseline.pkl"

        if self.baseline_model:
            self.baseline_model.save(baseline_path)

        # Save forecaster state
        state = {
            'baseline_type': self.baseline_type,
            'fitted_region': self.fitted_region,
            'feature_names': self.feature_names,
            'feature_weights': self.feature_weights,
            'historical_data': self.historical_data,
            'baseline_path': str(baseline_path),
        }

        with open(path, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved forecaster to {path}")

    @classmethod
    def load(cls, path: Path) -> 'RiskForecaster':
        """Load forecaster from disk.

        Args:
            path: Path to saved forecaster

        Returns:
            Loaded RiskForecaster instance

        Raises:
            FileNotFoundError: If forecaster file not found
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Forecaster file not found: {path}")

        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Create instance
        instance = cls(baseline_type=state['baseline_type'])

        # Restore state
        instance.fitted_region = state['fitted_region']
        instance.feature_names = state['feature_names']
        instance.feature_weights = state['feature_weights']
        instance.historical_data = state['historical_data']

        # Load baseline model
        baseline_path = Path(state['baseline_path'])
        if baseline_path.exists():
            instance.baseline_model = BaselineForecaster.load(baseline_path)

        logger.info(f"Loaded forecaster from {path}")

        return instance

    def _pivot_to_wide(self, long_data: pd.DataFrame) -> pd.DataFrame:
        """Convert long-format data to wide format.

        Args:
            long_data: DataFrame with columns: date, feature_name, value

        Returns:
            Wide-format DataFrame with features as columns
        """
        # Pivot: rows=date, columns=feature_name, values=value
        wide = long_data.pivot_table(
            index='date',
            columns='feature_name',
            values='value',
            aggfunc='mean'  # Handle potential duplicates
        )

        # Ensure datetime index
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()

        # Handle missing values
        if wide.isna().any().any():
            n_missing = wide.isna().sum().sum()
            logger.warning(f"Found {n_missing} missing values, forward-filling")
            wide = wide.fillna(method='ffill').fillna(method='bfill')

        return wide

    def _create_synthetic_risk_score(self, data: pd.DataFrame) -> pd.Series:
        """Create synthetic risk score from features.

        Uses normalized weighted sum of features. Features with higher
        variance get higher weights (assuming they're more informative).

        Args:
            data: Wide-format DataFrame with features

        Returns:
            Risk score series
        """
        logger.info("Creating synthetic risk score from features")

        # Normalize each feature to [0, 1]
        normalized = pd.DataFrame()

        for col in data.columns:
            col_min = data[col].min()
            col_max = data[col].max()

            if col_max > col_min:
                normalized[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                normalized[col] = 0.5  # Constant feature

        # Weight by variance (features with more variance are more informative)
        variances = normalized.var()
        total_var = variances.sum()

        if total_var > 0:
            weights = variances / total_var
        else:
            # Equal weights if all features are constant
            weights = pd.Series(1.0 / len(normalized.columns), index=normalized.columns)

        # Compute weighted sum
        risk_score = (normalized * weights).sum(axis=1)

        logger.info(
            f"Created risk score (mean={risk_score.mean():.3f}, "
            f"std={risk_score.std():.3f})"
        )

        return risk_score

    def _compute_feature_weights(self, data: pd.DataFrame, target_col: str):
        """Compute feature importance weights.

        Uses correlation with target as a simple importance measure.

        Args:
            data: Wide-format DataFrame
            target_col: Target column name
        """
        if target_col not in data.columns:
            logger.warning(f"Target column '{target_col}' not found")
            return

        # Compute correlations with target
        target = data[target_col]
        correlations = {}

        for col in self.feature_names:
            if col in data.columns:
                corr = data[col].corr(target)
                correlations[col] = abs(corr)  # Use absolute correlation

        # Normalize to sum to 1
        total = sum(correlations.values())

        if total > 0:
            self.feature_weights = {
                k: v / total for k, v in correlations.items()
            }
        else:
            # Equal weights if no correlation
            self.feature_weights = {
                k: 1.0 / len(correlations) for k in correlations
            }

        logger.debug(f"Computed feature weights for {len(self.feature_weights)} features")

    def _identify_drivers(self, top_k: int = 5) -> List[str]:
        """Identify top feature drivers.

        Args:
            top_k: Number of top features to return

        Returns:
            List of top feature names sorted by importance
        """
        if not self.feature_weights:
            return []

        # Sort by weight descending
        sorted_features = sorted(
            self.feature_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [name for name, _ in sorted_features[:top_k]]

    def _compute_confidence(self, risk_scores: pd.DataFrame) -> float:
        """Compute forecast confidence based on prediction uncertainty.

        Higher uncertainty (wider intervals) leads to lower confidence.

        Args:
            risk_scores: DataFrame with score, lower, upper columns

        Returns:
            Confidence score between 0 and 1
        """
        # Compute average relative interval width
        avg_score = risk_scores['score'].mean()

        if avg_score > 0:
            interval_widths = (risk_scores['upper'] - risk_scores['lower']) / avg_score
            avg_relative_width = interval_widths.mean()

            # Convert to confidence: narrower intervals = higher confidence
            # Use sigmoid-like transformation
            confidence = 1.0 / (1.0 + avg_relative_width)
        else:
            # Default to medium confidence if score is zero
            confidence = 0.5

        return float(np.clip(confidence, 0.0, 1.0))
