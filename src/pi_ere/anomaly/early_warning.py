"""Early warning system for converting anomalies into actionable alerts.

Analyzes detected anomalies, forecasts, and threshold breaches to generate
prioritized alerts with recommended actions for operational risk management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from pi_ere.anomaly.detector import Anomaly, AnomalyDetector
    from pi_ere.models.forecaster import RiskForecast


@dataclass
class Alert:
    """Early warning alert with severity, type, and recommendations.

    Attributes:
        level: Alert severity ('watch', 'warning', 'critical')
        alert_type: Type of alert ('escalation', 'anomaly', 'threshold', 'trend')
        region: Region identifier
        message: Human-readable alert message
        confidence: Alert confidence score (0-1)
        timestamp: When alert was generated
        recommended_actions: List of recommended actions
        related_anomalies: List of related anomalies
        metadata: Additional metadata
    """
    level: str
    alert_type: str
    region: str
    message: str
    confidence: float
    timestamp: datetime
    recommended_actions: List[str] = field(default_factory=list)
    related_anomalies: List['Anomaly'] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AlertSummary:
    """Summary statistics for a collection of alerts.

    Attributes:
        total_alerts: Total number of alerts
        by_level: Count of alerts by severity level
        by_type: Count of alerts by type
        by_region: Count of alerts by region
        top_concerns: List of top concern messages
        generated_at: Timestamp when summary was generated
    """
    total_alerts: int
    by_level: Dict[str, int]
    by_type: Dict[str, int]
    by_region: Dict[str, int]
    top_concerns: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class EarlyWarningSystem:
    """Generate and manage early warning alerts from anomalies and forecasts.

    Converts detected anomalies, forecast trajectories, and threshold breaches
    into actionable alerts with severity levels and recommended actions.
    """

    def __init__(
        self,
        detector: Optional['AnomalyDetector'] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize early warning system.

        Args:
            detector: Optional AnomalyDetector instance for anomaly detection
            thresholds: Optional severity thresholds for alert levels
                       Default: {'watch': 0.3, 'warning': 0.5, 'critical': 0.8}
        """
        self.detector = detector
        self.thresholds = thresholds or {
            'watch': 0.3,
            'warning': 0.5,
            'critical': 0.8
        }

        logger.info(
            f"Initialized EarlyWarningSystem "
            f"(has_detector={detector is not None}, "
            f"thresholds={self.thresholds})"
        )

    def analyze(
        self,
        region: str,
        data: pd.DataFrame,
        embedding: Optional[np.ndarray] = None,
        forecast: Optional['RiskForecast'] = None
    ) -> List[Alert]:
        """Generate alerts from anomalies, forecasts, and threshold breaches.

        Args:
            region: Region identifier
            data: Historical data for analysis
            embedding: Optional region embedding vector
            forecast: Optional risk forecast

        Returns:
            List of generated alerts
        """
        logger.info(f"Analyzing region '{region}' for early warnings")

        alerts = []
        timestamp = datetime.now()

        # 1. Detect anomalies using detector
        if self.detector is not None and not data.empty:
            anomalies = self.detector.detect_anomalies(data, region=region)

            for anomaly in anomalies:
                alert = self._create_anomaly_alert(
                    anomaly, region, timestamp
                )
                alerts.append(alert)

            logger.debug(f"Generated {len(anomalies)} anomaly-based alerts")

        # 2. Analyze forecast trajectory
        if forecast is not None:
            forecast_alerts = self._analyze_forecast_trajectory(
                forecast, region, timestamp
            )
            alerts.extend(forecast_alerts)

            logger.debug(f"Generated {len(forecast_alerts)} forecast-based alerts")

        # 3. Check threshold breaches
        if not data.empty:
            threshold_alerts = self._check_threshold_breaches(
                data, region, timestamp
            )
            alerts.extend(threshold_alerts)

            logger.debug(f"Generated {len(threshold_alerts)} threshold-based alerts")

        logger.info(f"Generated {len(alerts)} total alerts for region '{region}'")

        return alerts

    def prioritize_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Sort alerts by urgency (level, then confidence).

        Args:
            alerts: List of alerts to prioritize

        Returns:
            Sorted list of alerts (most urgent first)
        """
        level_priority = {'critical': 0, 'warning': 1, 'watch': 2}

        sorted_alerts = sorted(
            alerts,
            key=lambda a: (level_priority.get(a.level, 3), -a.confidence)
        )

        logger.debug(f"Prioritized {len(alerts)} alerts")

        return sorted_alerts

    def get_alert_summary(self, alerts: List[Alert]) -> AlertSummary:
        """Generate summary statistics for alerts.

        Args:
            alerts: List of alerts

        Returns:
            AlertSummary with statistics
        """
        by_level = {}
        by_type = {}
        by_region = {}

        for alert in alerts:
            by_level[alert.level] = by_level.get(alert.level, 0) + 1
            by_type[alert.alert_type] = by_type.get(alert.alert_type, 0) + 1
            by_region[alert.region] = by_region.get(alert.region, 0) + 1

        # Extract top concerns (critical and warning messages)
        top_concerns = [
            alert.message
            for alert in alerts
            if alert.level in ['critical', 'warning']
        ][:5]  # Top 5

        summary = AlertSummary(
            total_alerts=len(alerts),
            by_level=by_level,
            by_type=by_type,
            by_region=by_region,
            top_concerns=top_concerns
        )

        logger.debug(f"Generated summary for {len(alerts)} alerts")

        return summary

    def generate_recommendations(self, alert: Alert) -> List[str]:
        """Generate recommended actions based on alert type.

        Args:
            alert: Alert to generate recommendations for

        Returns:
            List of recommended action items
        """
        recommendations_by_type = {
            'escalation': [
                "Monitor situation closely",
                "Review contingency plans",
                "Brief stakeholders"
            ],
            'anomaly': [
                "Investigate data source",
                "Cross-reference with local reports"
            ],
            'threshold': [
                "Review risk exposure",
                "Consider operational adjustments"
            ],
            'trend': [
                "Analyze historical patterns",
                "Update risk models",
                "Prepare mitigation strategies"
            ]
        }

        return recommendations_by_type.get(alert.alert_type, [
            "Review alert details",
            "Consult subject matter experts"
        ])

    def _create_anomaly_alert(
        self,
        anomaly: 'Anomaly',
        region: str,
        timestamp: datetime
    ) -> Alert:
        """Create alert from detected anomaly.

        Args:
            anomaly: Detected anomaly
            region: Region identifier
            timestamp: Alert timestamp

        Returns:
            Alert object
        """
        # Determine alert level based on severity
        severity = anomaly.severity
        if severity >= self.thresholds['critical']:
            level = 'critical'
        elif severity >= self.thresholds['warning']:
            level = 'warning'
        else:
            level = 'watch'

        # Determine alert type
        alert_type = anomaly.anomaly_type if hasattr(anomaly, 'anomaly_type') else 'anomaly'

        message = f"Anomaly detected: {anomaly.description if hasattr(anomaly, 'description') else 'Unusual pattern observed'}"

        alert = Alert(
            level=level,
            alert_type=alert_type,
            region=region,
            message=message,
            confidence=anomaly.confidence if hasattr(anomaly, 'confidence') else severity,
            timestamp=timestamp,
            related_anomalies=[anomaly],
            metadata={'anomaly_id': getattr(anomaly, 'id', None)}
        )

        alert.recommended_actions = self.generate_recommendations(alert)

        return alert

    def _analyze_forecast_trajectory(
        self,
        forecast: 'RiskForecast',
        region: str,
        timestamp: datetime
    ) -> List[Alert]:
        """Analyze forecast trajectory for concerning trends.

        Args:
            forecast: Risk forecast
            region: Region identifier
            timestamp: Alert timestamp

        Returns:
            List of alerts
        """
        alerts = []

        risk_scores = forecast.risk_scores['score'].values

        # Check if risk is increasing
        if len(risk_scores) >= 2:
            trend = np.polyfit(range(len(risk_scores)), risk_scores, deg=1)[0]

            if trend > 0.01:  # Increasing trend threshold
                severity = min(abs(trend) * 10, 1.0)  # Scale to 0-1

                if severity >= self.thresholds['warning']:
                    level = 'critical' if severity >= self.thresholds['critical'] else 'warning'

                    alert = Alert(
                        level=level,
                        alert_type='trend',
                        region=region,
                        message=f"Risk forecast shows increasing trend (slope={trend:.4f})",
                        confidence=forecast.confidence,
                        timestamp=timestamp,
                        metadata={'forecast_horizon': forecast.horizon_days}
                    )

                    alert.recommended_actions = self.generate_recommendations(alert)
                    alerts.append(alert)

        return alerts

    def _check_threshold_breaches(
        self,
        data: pd.DataFrame,
        region: str,
        timestamp: datetime
    ) -> List[Alert]:
        """Check for threshold breaches on key indicators.

        Args:
            data: Historical data
            region: Region identifier
            timestamp: Alert timestamp

        Returns:
            List of alerts
        """
        alerts = []

        # Check if data has value column for threshold checking
        if 'value' in data.columns:
            recent_values = data['value'].tail(5)

            if len(recent_values) > 0:
                avg_value = recent_values.mean()

                # Normalize to 0-1 range if needed
                if avg_value > 1.0:
                    normalized_value = min(avg_value / 100.0, 1.0)
                else:
                    normalized_value = abs(avg_value)

                if normalized_value >= self.thresholds['warning']:
                    level = 'critical' if normalized_value >= self.thresholds['critical'] else 'warning'

                    alert = Alert(
                        level=level,
                        alert_type='threshold',
                        region=region,
                        message=f"Key indicator exceeds {level} threshold (value={avg_value:.2f})",
                        confidence=0.9,  # High confidence for threshold breaches
                        timestamp=timestamp,
                        metadata={'threshold_value': normalized_value}
                    )

                    alert.recommended_actions = self.generate_recommendations(alert)
                    alerts.append(alert)

        return alerts
