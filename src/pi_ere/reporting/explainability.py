"""Feature attribution and narrative generation for risk forecasts.

This module provides explainability tools for understanding risk forecasts,
including feature importance analysis, driving factor identification, and
narrative generation with optional LLM integration.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from pi_ere.models.forecaster import RiskForecast


@dataclass
class Driver:
    """Represents a key factor driving risk forecast.

    Attributes:
        feature: Feature name
        contribution: Importance score (0-1)
        direction: Trend direction ('increasing', 'decreasing', 'stable')
        description: Human-readable description
    """

    feature: str
    contribution: float  # 0-1 importance score
    direction: str  # 'increasing', 'decreasing', 'stable'
    description: str


@dataclass
class Explanation:
    """Complete explanation for a risk forecast.

    Attributes:
        top_drivers: List of top driving factors
        feature_weights: Dictionary of feature importance scores
        historical_context: Contextual information about historical patterns
        narrative: Human-readable explanation narrative
        analogues_used: List of similar historical cases used
        confidence: Overall explanation confidence (0-1)
        metadata: Additional metadata
    """

    top_drivers: List[Driver]
    feature_weights: Dict[str, float]
    historical_context: str
    narrative: str
    analogues_used: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)


class ExplainabilityEngine:
    """Feature attribution and narrative generation for risk forecasts.

    Provides methods to explain risk forecasts through feature importance,
    driving factor identification, and narrative generation. Supports
    optional LLM integration for enhanced narrative generation.
    """

    # Narrative templates for different risk levels
    TEMPLATES = {
        "high": (
            "Risk levels in {region} are elevated with {confidence:.0%} confidence. "
            "The primary driver is {top_driver}, which has been {direction} over the past 30 days. "
            "{historical_context}"
        ),
        "moderate": (
            "Moderate risk conditions persist in {region}. "
            "Key factors include {top_driver}, showing {direction} trends. "
            "{historical_context}"
        ),
        "low": (
            "Risk levels in {region} remain low with {confidence:.0%} confidence. "
            "Dominant factor {top_driver} exhibits {direction} patterns. "
            "{historical_context}"
        ),
    }

    def __init__(self, llm_backend: Optional[Any] = None):
        """Initialize explainability engine.

        Args:
            llm_backend: Optional LLM client for enhanced narrative generation
        """
        self.llm_backend = llm_backend
        logger.info(
            f"Initialized ExplainabilityEngine "
            f"(llm_enabled={llm_backend is not None})"
        )

    def explain_forecast(
        self,
        forecast: "RiskForecast",
        data: pd.DataFrame,
        analogues: List = None,
    ) -> Explanation:
        """Generate comprehensive explanation for a risk forecast.

        Args:
            forecast: RiskForecast object to explain
            data: Historical data DataFrame (wide format)
            analogues: Optional list of analogous historical cases

        Returns:
            Explanation object with feature attribution and narrative

        Raises:
            ValueError: If data format is invalid
        """
        logger.info(f"Generating explanation for forecast in region '{forecast.region}'")

        # Validate data format
        if data.empty:
            raise ValueError("Data cannot be empty")

        # Extract target series from forecast risk scores
        target = forecast.risk_scores["score"]

        # Get feature importance
        feature_weights = self.get_feature_importance(data, target)

        # Find driving factors
        drivers = self.find_driving_factors(data, anomalies=None)

        # Generate historical context
        historical_context = self._generate_historical_context(data, forecast)

        # Determine risk level for template selection
        avg_risk = forecast.risk_scores["score"].mean()
        risk_level = self._classify_risk_level(avg_risk)

        # Generate narrative
        context = {
            "region": forecast.region,
            "risk_level": risk_level,
            "top_driver": drivers[0].feature if drivers else "unknown",
            "direction": drivers[0].direction if drivers else "stable",
            "confidence": forecast.confidence,
            "historical_context": historical_context,
        }

        narrative = self.generate_narrative(context)

        # Create explanation
        explanation = Explanation(
            top_drivers=drivers[:5],  # Top 5 drivers
            feature_weights=feature_weights,
            historical_context=historical_context,
            narrative=narrative,
            analogues_used=analogues if analogues else [],
            confidence=forecast.confidence,
            metadata={
                "forecast_date": forecast.forecast_date.isoformat(),
                "horizon_days": forecast.horizon_days,
                "avg_risk_score": float(avg_risk),
                "risk_level": risk_level,
                "n_drivers": len(drivers),
            },
        )

        logger.info(
            f"Explanation generated: {len(drivers)} drivers, "
            f"risk_level={risk_level}, confidence={forecast.confidence:.3f}"
        )

        return explanation

    def get_feature_importance(
        self, data: pd.DataFrame, target: pd.Series
    ) -> Dict[str, float]:
        """Calculate feature importance using correlation-based method.

        Args:
            data: DataFrame with features as columns
            target: Target series to compute importance against

        Returns:
            Dictionary mapping feature names to importance scores (0-1)

        Raises:
            ValueError: If target length doesn't match data
        """
        logger.debug(f"Computing feature importance for {len(data.columns)} features")

        # Align target with data if needed
        if len(target) != len(data):
            logger.warning(
                f"Target length ({len(target)}) doesn't match data ({len(data)}), "
                "using first {len(data)} values"
            )
            target = target.iloc[: len(data)]

        # Compute absolute correlations with target
        correlations = {}

        for col in data.columns:
            if col in ["date", "Date", "timestamp", "Timestamp"]:
                continue

            # Skip if all values are NaN
            if data[col].isna().all():
                continue

            # Compute correlation
            corr = data[col].corr(target)

            if not pd.isna(corr):
                correlations[col] = abs(corr)
            else:
                correlations[col] = 0.0

        # Normalize to sum to 1
        total = sum(correlations.values())

        if total > 0:
            feature_weights = {k: v / total for k, v in correlations.items()}
        else:
            # Equal weights if no correlation
            n_features = len(correlations)
            feature_weights = {k: 1.0 / n_features for k in correlations}

        logger.debug(
            f"Computed {len(feature_weights)} feature weights "
            f"(max={max(feature_weights.values()):.3f})"
        )

        return feature_weights

    def find_driving_factors(
        self, data: pd.DataFrame, anomalies: List = None
    ) -> List[Driver]:
        """Identify features with significant recent changes.

        Args:
            data: DataFrame with features as columns and datetime index
            anomalies: Optional list of detected anomalies

        Returns:
            List of Driver objects sorted by contribution
        """
        logger.debug("Identifying driving factors from recent changes")

        drivers = []

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data does not have datetime index, using row numbers")
            data = data.copy()
            data.index = pd.RangeIndex(len(data))

        # Define lookback window (30 days or 30% of data)
        lookback_window = min(30, max(int(len(data) * 0.3), 1))

        for col in data.columns:
            if col in ["date", "Date", "timestamp", "Timestamp"]:
                continue

            if data[col].isna().all():
                continue

            # Get recent and historical values
            recent_values = data[col].iloc[-lookback_window:]
            historical_values = data[col].iloc[:-lookback_window] if len(data) > lookback_window else data[col]

            if len(recent_values) == 0 or len(historical_values) == 0:
                continue

            # Compute statistics
            recent_mean = recent_values.mean()
            historical_mean = historical_values.mean()
            historical_std = historical_values.std()

            # Skip if no variation
            if historical_std == 0 or pd.isna(historical_std):
                continue

            # Calculate change magnitude (z-score)
            change_magnitude = abs(recent_mean - historical_mean) / historical_std

            # Determine direction
            if recent_mean > historical_mean * 1.1:
                direction = "increasing"
            elif recent_mean < historical_mean * 0.9:
                direction = "decreasing"
            else:
                direction = "stable"

            # Compute contribution score (normalized change magnitude)
            contribution = min(change_magnitude / 3.0, 1.0)  # Cap at z-score of 3

            # Generate description
            pct_change = (
                ((recent_mean - historical_mean) / abs(historical_mean) * 100)
                if historical_mean != 0
                else 0.0
            )

            description = (
                f"{col} has changed {direction} by {abs(pct_change):.1f}% "
                f"compared to historical baseline (z-score: {change_magnitude:.2f})"
            )

            drivers.append(
                Driver(
                    feature=col,
                    contribution=contribution,
                    direction=direction,
                    description=description,
                )
            )

        # Sort by contribution descending
        drivers.sort(key=lambda d: d.contribution, reverse=True)

        logger.debug(
            f"Identified {len(drivers)} driving factors "
            f"(top: {drivers[0].feature if drivers else 'none'})"
        )

        return drivers

    def generate_narrative(self, context: Dict) -> str:
        """Generate human-readable narrative from context.

        Uses template-based generation by default, with optional LLM
        enhancement if backend is configured.

        Args:
            context: Dictionary with narrative context including:
                    - region: Region name
                    - risk_level: Risk level ('high', 'moderate', 'low')
                    - top_driver: Primary driving feature
                    - direction: Trend direction
                    - confidence: Confidence score
                    - historical_context: Historical context string

        Returns:
            Generated narrative string
        """
        logger.debug(
            f"Generating narrative for risk_level={context.get('risk_level', 'unknown')}"
        )

        # Use LLM if available
        if self.llm_backend is not None:
            return self._generate_llm_narrative(context)

        # Otherwise use template-based generation
        risk_level = context.get("risk_level", "moderate")
        template = self.TEMPLATES.get(risk_level, self.TEMPLATES["moderate"])

        try:
            narrative = template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context key for narrative: {e}, using fallback")
            narrative = (
                f"Risk analysis for {context.get('region', 'unknown region')} "
                f"shows {risk_level} risk conditions with "
                f"{context.get('confidence', 0.5):.0%} confidence."
            )

        logger.debug(f"Generated narrative ({len(narrative)} chars)")
        return narrative

    def set_llm_backend(self, client: Any) -> None:
        """Set LLM backend for enhanced narrative generation.

        Args:
            client: LLM client instance (e.g., OpenAI, Anthropic)
        """
        self.llm_backend = client
        logger.info("LLM backend configured for narrative generation")

    def _generate_llm_narrative(self, context: Dict) -> str:
        """Generate narrative using LLM backend.

        Args:
            context: Narrative context dictionary

        Returns:
            LLM-generated narrative string
        """
        logger.debug("Generating narrative using LLM backend")

        # Construct prompt
        prompt = (
            f"Generate a concise risk analysis narrative for {context.get('region')} "
            f"with {context.get('risk_level')} risk level. "
            f"Primary driver: {context.get('top_driver')} ({context.get('direction')}). "
            f"Confidence: {context.get('confidence', 0):.0%}. "
            f"Context: {context.get('historical_context', 'No additional context.')}"
        )

        try:
            # This is a placeholder for actual LLM integration
            # Different LLM backends will have different APIs
            response = self.llm_backend.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM narrative generation failed: {e}, falling back to template")
            # Fall back to template-based generation
            risk_level = context.get("risk_level", "moderate")
            template = self.TEMPLATES.get(risk_level, self.TEMPLATES["moderate"])
            return template.format(**context)

    def _generate_historical_context(
        self, data: pd.DataFrame, forecast: "RiskForecast"
    ) -> str:
        """Generate historical context summary.

        Args:
            data: Historical data DataFrame
            forecast: RiskForecast object

        Returns:
            Historical context string
        """
        # Calculate basic statistics
        n_points = len(data)
        time_span_days = (
            (data.index[-1] - data.index[0]).days
            if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1
            else 0
        )

        context = (
            f"Analysis based on {n_points} historical observations "
            f"spanning {time_span_days} days. "
        )

        # Add driver information if available
        if forecast.drivers:
            context += f"Historical patterns show {len(forecast.drivers)} key drivers. "

        return context

    def _classify_risk_level(self, avg_risk: float) -> str:
        """Classify average risk score into categorical level.

        Args:
            avg_risk: Average risk score

        Returns:
            Risk level string ('high', 'moderate', 'low')
        """
        if avg_risk > 0.7:
            return "high"
        elif avg_risk > 0.4:
            return "moderate"
        else:
            return "low"
