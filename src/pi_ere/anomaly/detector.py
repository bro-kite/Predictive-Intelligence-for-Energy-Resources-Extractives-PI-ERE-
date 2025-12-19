"""Multi-method anomaly detection for risk signals in PI-ERE.

This module provides anomaly detection using multiple methods:
- Residual-based detection (statistical deviation from baseline)
- Isolation Forest (multi-feature anomaly detection)
- Embedding drift (semantic shift detection)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class Anomaly:
    """Represents a detected anomaly with metadata."""

    anomaly_type: str  # 'residual', 'isolation_forest', 'embedding_drift'
    feature: str
    severity: float  # 0-1 scale
    timestamp: datetime
    description: str
    raw_score: float
    metadata: Dict = field(default_factory=dict)


class AnomalyDetector:
    """Multi-method anomaly detection for risk signals.

    Supports three detection methods:
    - residual: Statistical deviation from baseline (>3 std)
    - isolation_forest: Sklearn IsolationForest on multi-feature space
    - embedding_drift: Cosine distance from mean historical embedding
    """

    def __init__(
        self,
        methods: List[str] = None,
        contamination: float = 0.1,
        residual_threshold: float = 3.0,
    ):
        """Initialize anomaly detector.

        Args:
            methods: List of detection methods to use.
                     Options: ['residual', 'isolation_forest', 'embedding_drift']
                     Default: ['residual', 'isolation_forest']
            contamination: Expected proportion of outliers (for IsolationForest)
            residual_threshold: Number of std deviations for residual method
        """
        if methods is None:
            methods = ["residual", "isolation_forest"]

        valid_methods = {"residual", "isolation_forest", "embedding_drift"}
        for method in methods:
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method '{method}'. Must be one of {valid_methods}"
                )

        self.methods = methods
        self.contamination = contamination
        self.residual_threshold = residual_threshold

        # Baseline statistics (populated during fit)
        self.baseline_stats: Dict[str, Dict] = {}
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.mean_embedding: Optional[np.ndarray] = None
        self.feature_columns: List[str] = []

        logger.info(
            f"Initialized AnomalyDetector with methods={methods}, "
            f"contamination={contamination}"
        )

    def fit(
        self,
        historical_data: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None,
    ):
        """Fit detector on historical data to establish baselines.

        Args:
            historical_data: DataFrame with columns [date, feature_name, value]
                            or wide format [date, feature1, feature2, ...]
            embeddings: Optional array of historical embeddings (n_samples, embedding_dim)
        """
        logger.info(f"Fitting anomaly detector on {len(historical_data)} samples")

        # Detect data format and prepare features
        df = historical_data.copy()

        if "feature_name" in df.columns and "value" in df.columns:
            # Long format - pivot to wide
            df = df.pivot_table(
                index="date", columns="feature_name", values="value"
            ).reset_index()
            logger.debug("Converted long format to wide format")

        # Extract feature columns (exclude date column)
        date_cols = ["date", "Date", "timestamp", "Timestamp"]
        self.feature_columns = [
            col for col in df.columns if col not in date_cols
        ]
        logger.info(f"Detected {len(self.feature_columns)} features")

        # Fit residual-based detection
        if "residual" in self.methods:
            self._fit_residual(df)

        # Fit isolation forest
        if "isolation_forest" in self.methods:
            self._fit_isolation_forest(df)

        # Fit embedding drift detection
        if "embedding_drift" in self.methods:
            self._fit_embedding_drift(embeddings)

        logger.info("Anomaly detector fitting complete")

    def _fit_residual(self, df: pd.DataFrame):
        """Fit residual-based detection (statistical baselines)."""
        logger.debug("Fitting residual-based detection")

        for feature in self.feature_columns:
            if feature not in df.columns:
                continue

            values = df[feature].dropna()
            self.baseline_stats[feature] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "count": len(values),
            }

        logger.debug(f"Computed baseline stats for {len(self.baseline_stats)} features")

    def _fit_isolation_forest(self, df: pd.DataFrame):
        """Fit IsolationForest on multi-feature space."""
        logger.debug("Fitting IsolationForest")

        # Prepare feature matrix
        X = df[self.feature_columns].fillna(0).values

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit isolation forest
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
        )
        self.isolation_forest.fit(X_scaled)

        logger.debug("IsolationForest fitted successfully")

    def _fit_embedding_drift(self, embeddings: Optional[np.ndarray]):
        """Fit embedding drift detection (compute mean embedding)."""
        if embeddings is None:
            logger.warning(
                "Embedding drift requested but no embeddings provided, skipping"
            )
            return

        logger.debug("Fitting embedding drift detection")

        # Compute mean embedding as baseline
        self.mean_embedding = embeddings.mean(axis=0)
        logger.debug(
            f"Computed mean embedding with dimension {self.mean_embedding.shape[0]}"
        )

    def detect(
        self,
        current_data: pd.DataFrame,
        current_embedding: Optional[np.ndarray] = None,
    ) -> List[Anomaly]:
        """Detect anomalies in current data using fitted baselines.

        Args:
            current_data: DataFrame with same format as training data
            current_embedding: Optional current embedding vector

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Prepare data
        df = current_data.copy()

        if "feature_name" in df.columns and "value" in df.columns:
            # Long format - pivot to wide
            df = df.pivot_table(
                index="date", columns="feature_name", values="value"
            ).reset_index()

        # Extract timestamp (use first row if multiple)
        timestamp = self._extract_timestamp(df)

        # Run residual detection
        if "residual" in self.methods:
            anomalies.extend(self._detect_residual(df, timestamp))

        # Run isolation forest detection
        if "isolation_forest" in self.methods:
            anomalies.extend(self._detect_isolation_forest(df, timestamp))

        # Run embedding drift detection
        if "embedding_drift" in self.methods:
            anomalies.extend(
                self._detect_embedding_drift(current_embedding, timestamp)
            )

        logger.info(f"Detected {len(anomalies)} anomalies across all methods")
        return anomalies

    def _extract_timestamp(self, df: pd.DataFrame) -> datetime:
        """Extract timestamp from dataframe."""
        date_cols = ["date", "Date", "timestamp", "Timestamp"]

        for col in date_cols:
            if col in df.columns:
                date_val = df[col].iloc[0]
                if isinstance(date_val, datetime):
                    return date_val
                return pd.to_datetime(date_val)

        # Default to current time
        return datetime.now()

    def _detect_residual(
        self, df: pd.DataFrame, timestamp: datetime
    ) -> List[Anomaly]:
        """Detect residual-based anomalies."""
        anomalies = []

        for feature in self.feature_columns:
            if feature not in df.columns or feature not in self.baseline_stats:
                continue

            value = df[feature].iloc[0]
            if pd.isna(value):
                continue

            stats = self.baseline_stats[feature]
            mean = stats["mean"]
            std = stats["std"]

            if std == 0:
                continue

            # Calculate z-score
            z_score = abs((value - mean) / std)

            if z_score > self.residual_threshold:
                severity = self.score_anomaly_value(z_score, method="residual")

                anomalies.append(
                    Anomaly(
                        anomaly_type="residual",
                        feature=feature,
                        severity=severity,
                        timestamp=timestamp,
                        description=f"{feature} deviates by {z_score:.2f} std from baseline",
                        raw_score=z_score,
                        metadata={
                            "value": float(value),
                            "baseline_mean": mean,
                            "baseline_std": std,
                            "z_score": float(z_score),
                        },
                    )
                )

        return anomalies

    def _detect_isolation_forest(
        self, df: pd.DataFrame, timestamp: datetime
    ) -> List[Anomaly]:
        """Detect anomalies using IsolationForest."""
        if self.isolation_forest is None or self.scaler is None:
            return []

        anomalies = []

        # Prepare feature matrix
        X = df[self.feature_columns].fillna(0).values
        X_scaled = self.scaler.transform(X)

        # Predict anomaly score
        predictions = self.isolation_forest.predict(X_scaled)
        scores = self.isolation_forest.score_samples(X_scaled)

        for i, (pred, score) in enumerate(zip(predictions, scores)):
            if pred == -1:  # Anomaly detected
                # Convert score to positive severity (more negative = more anomalous)
                raw_score = abs(score)
                severity = self.score_anomaly_value(raw_score, method="isolation_forest")

                anomalies.append(
                    Anomaly(
                        anomaly_type="isolation_forest",
                        feature="multi_feature",
                        severity=severity,
                        timestamp=timestamp,
                        description=f"Multi-feature anomaly detected (score={score:.3f})",
                        raw_score=raw_score,
                        metadata={
                            "isolation_score": float(score),
                            "sample_index": i,
                        },
                    )
                )

        return anomalies

    def _detect_embedding_drift(
        self, current_embedding: Optional[np.ndarray], timestamp: datetime
    ) -> List[Anomaly]:
        """Detect embedding drift using cosine distance."""
        if self.mean_embedding is None or current_embedding is None:
            return []

        anomalies = []

        # Compute cosine distance
        cosine_sim = np.dot(current_embedding, self.mean_embedding) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(self.mean_embedding)
        )
        cosine_distance = 1 - cosine_sim

        # Threshold for drift (configurable)
        drift_threshold = 0.3

        if cosine_distance > drift_threshold:
            severity = self.score_anomaly_value(
                cosine_distance, method="embedding_drift"
            )

            anomalies.append(
                Anomaly(
                    anomaly_type="embedding_drift",
                    feature="embedding",
                    severity=severity,
                    timestamp=timestamp,
                    description=f"Embedding drift detected (distance={cosine_distance:.3f})",
                    raw_score=float(cosine_distance),
                    metadata={
                        "cosine_distance": float(cosine_distance),
                        "cosine_similarity": float(cosine_sim),
                    },
                )
            )

        return anomalies

    def score_anomaly(self, anomaly: Anomaly) -> float:
        """Score anomaly severity (0-1 scale).

        Args:
            anomaly: Anomaly object to score

        Returns:
            Severity score between 0 and 1
        """
        return anomaly.severity

    def score_anomaly_value(self, raw_score: float, method: str) -> float:
        """Convert raw anomaly score to 0-1 severity using sigmoid.

        Args:
            raw_score: Raw anomaly score from detection method
            method: Detection method name

        Returns:
            Normalized severity score (0-1)
        """
        if method == "residual":
            # Map z-scores to 0-1 using sigmoid (z=3 -> 0.5, z=6 -> 0.9)
            return 1 / (1 + np.exp(-(raw_score - 3)))

        elif method == "isolation_forest":
            # Isolation forest scores are negative, more negative = more anomalous
            # Map to 0-1 using sigmoid
            return 1 / (1 + np.exp(-5 * raw_score))

        elif method == "embedding_drift":
            # Cosine distance is 0-2, but typically 0-1
            # Linear mapping with clipping
            return min(raw_score / 0.5, 1.0)

        else:
            # Default: clip to 0-1 range
            return min(max(raw_score, 0.0), 1.0)

    def get_baseline_stats(self) -> Dict[str, Dict]:
        """Get fitted baseline statistics.

        Returns:
            Dictionary of feature statistics and model info
        """
        stats = {
            "features": self.baseline_stats,
            "methods": self.methods,
            "contamination": self.contamination,
            "residual_threshold": self.residual_threshold,
        }

        if self.isolation_forest is not None:
            stats["isolation_forest"] = {
                "n_estimators": self.isolation_forest.n_estimators,
                "fitted": True,
            }

        if self.mean_embedding is not None:
            stats["embedding"] = {
                "dimension": self.mean_embedding.shape[0],
                "fitted": True,
            }

        return stats
