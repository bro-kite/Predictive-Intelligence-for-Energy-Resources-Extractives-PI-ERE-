# PI-ERE Phase 1: End-to-End Implementation Design

**Date:** 2025-12-19
**Branch:** `feature/phase1-forecasting-and-similarity`
**Goal:** Complete all missing modules to achieve end-to-end risk forecasting pipeline

## Overview

This design covers implementation of 7 missing modules to complete the PI-ERE pipeline:
- Data → Embeddings → Forecasting → Anomaly Detection → Alerts → Reporting

## Architecture

```
DATA LAYER (Done)              EMBEDDINGS (Done)
     │                              │
     └──────────┬──────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │   models/baseline.py   │  ← Prophet/ARIMA forecasts
    │   models/forecaster.py │  ← Unified risk scoring API
    └───────────┬───────────┘
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│ vector_db/   │  │ anomaly/         │
│ similarity.py│  │ detector.py      │
└──────┬───────┘  │ early_warning.py │
       │          └────────┬─────────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
    ┌────────────────────────┐
    │ reporting/             │
    │ explainability.py      │
    │ visualization.py       │
    └────────────────────────┘
```

## Module Specifications

### Wave 1: Models Layer

#### `models/baseline.py` - BaselineForecaster

Purpose: Traditional time-series forecasting as baseline and fallback.

```python
class BaselineForecaster:
    """ARIMA/Prophet baseline for risk indicator forecasting."""

    def __init__(self, model_type: str = 'prophet')
    def fit(self, series: pd.Series) -> self
    def predict(self, horizon: int = 180) -> pd.DataFrame  # days
    def get_prediction_intervals(self, confidence: float = 0.95) -> pd.DataFrame
    def save(self, path: Path) / load(cls, path: Path)
```

Key features:
- Supports both Prophet (default) and ARIMA
- Returns point forecasts + prediction intervals
- Handles missing data gracefully

#### `models/forecaster.py` - RiskForecaster

Purpose: Unified interface combining embeddings + baseline into risk scores.

```python
@dataclass
class RiskForecast:
    region: str
    forecast_date: datetime
    horizon_days: int
    risk_scores: pd.DataFrame  # date, score, lower, upper
    drivers: List[str]
    confidence: float

class RiskForecaster:
    """Main forecasting interface for operational risk."""

    def __init__(self, ts_embedder: TimeSeriesEmbedder, text_embedder: TextEmbedder)
    def fit(self, harmonized_data: pd.DataFrame, target_col: str = 'risk_score')
    def predict_risk(self, region: str, horizon_months: int = 6) -> RiskForecast
    def get_feature_importance(self) -> Dict[str, float]
```

### Wave 2: Vector DB & Anomaly Layer

#### `vector_db/similarity.py` - SimilaritySearch

Purpose: FAISS-based similarity search for historical analogues.

```python
@dataclass
class SearchResult:
    id: str
    score: float
    metadata: Dict

@dataclass
class Analogue:
    region: str
    start_date: datetime
    end_date: datetime
    similarity_score: float
    outcome: str  # what happened next
    key_features: List[str]

class SimilaritySearch:
    """FAISS-based similarity search for historical analogues."""

    def __init__(self, embedding_dim: int = 320, index_type: str = 'flat')
    def add_embeddings(self, ids: List[str], embeddings: np.ndarray, metadata: pd.DataFrame)
    def search(self, query: np.ndarray, top_k: int = 10) -> List[SearchResult]
    def find_analogues(self, region: str, current_embedding: np.ndarray,
                       exclude_recent_days: int = 90) -> List[Analogue]
    def find_similar_regions(self, region: str, date: datetime) -> List[SearchResult]
    def save_index(self, path: Path)
    def load_index(cls, path: Path)
```

Backend: FAISS with flat index (exact search, suitable for <1M vectors).

#### `anomaly/detector.py` - AnomalyDetector

Purpose: Multi-method anomaly detection for risk signals.

```python
@dataclass
class Anomaly:
    anomaly_type: str  # 'residual', 'isolation_forest', 'embedding_drift'
    feature: str
    severity: float  # 0-1
    timestamp: datetime
    description: str
    raw_score: float

class AnomalyDetector:
    """Multi-method anomaly detection for risk signals."""

    def __init__(self, methods: List[str] = ['residual', 'isolation_forest', 'embedding_drift'])
    def fit(self, historical_data: pd.DataFrame, embeddings: Optional[np.ndarray] = None)
    def detect(self, current_data: pd.DataFrame,
               current_embedding: Optional[np.ndarray] = None) -> List[Anomaly]
    def score_anomaly(self, anomaly: Anomaly) -> float
```

Methods:
- Forecast residual-based (MAE/MSE thresholds)
- Isolation Forest on multi-feature space
- Embedding drift (cosine distance from baseline)

#### `anomaly/early_warning.py` - EarlyWarningSystem

Purpose: Convert anomalies + forecasts into actionable alerts.

```python
@dataclass
class Alert:
    level: str  # 'watch', 'warning', 'critical'
    alert_type: str  # 'escalation', 'anomaly', 'threshold'
    region: str
    message: str
    confidence: float
    timestamp: datetime
    recommended_actions: List[str]
    related_anomalies: List[Anomaly]

@dataclass
class AlertSummary:
    total_alerts: int
    by_level: Dict[str, int]
    by_region: Dict[str, int]
    top_concerns: List[str]

class EarlyWarningSystem:
    """Generate and manage early warning alerts."""

    def __init__(self, detector: AnomalyDetector, forecaster: Optional[RiskForecaster] = None)
    def analyze(self, region: str, data: pd.DataFrame,
                embedding: Optional[np.ndarray] = None) -> List[Alert]
    def prioritize_alerts(self, alerts: List[Alert]) -> List[Alert]
    def get_alert_summary(self, alerts: List[Alert]) -> AlertSummary
```

### Wave 3: Reporting Layer

#### `reporting/explainability.py` - ExplainabilityEngine

Purpose: Explain why risk scores are elevated and what drives predictions.

```python
@dataclass
class Driver:
    feature: str
    contribution: float
    direction: str  # 'increasing', 'decreasing'
    description: str

@dataclass
class Explanation:
    top_drivers: List[Driver]
    feature_weights: Dict[str, float]
    historical_context: str
    narrative: str
    analogues_used: List[str]

class ExplainabilityEngine:
    """Feature attribution and narrative generation."""

    def __init__(self, llm_backend: Optional[Any] = None)
    def explain_forecast(self, forecast: RiskForecast, data: pd.DataFrame) -> Explanation
    def get_feature_importance(self, model: Any, data: pd.DataFrame) -> Dict[str, float]
    def find_driving_factors(self, region: str, anomalies: List[Anomaly]) -> List[Driver]
    def generate_narrative(self, context: Dict) -> str  # Template-based, LLM-ready
    def set_llm_backend(self, client: Any) -> None
```

#### `reporting/visualization.py` - RiskVisualizer

Purpose: Generate charts and dashboards for risk assessment.

```python
class RiskVisualizer:
    """Visualization tools for risk forecasts and alerts."""

    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 8))
    def plot_risk_trajectory(self, forecast: RiskForecast, ax=None) -> plt.Figure
    def plot_feature_importance(self, explanation: Explanation) -> plt.Figure
    def plot_analogues(self, analogues: List[Analogue],
                       current_embedding: np.ndarray,
                       method: str = 'umap') -> plt.Figure
    def plot_alert_timeline(self, alerts: List[Alert]) -> plt.Figure
    def plot_anomaly_scores(self, anomalies: List[Anomaly]) -> plt.Figure
    def create_region_dashboard(self, region: str, forecast: RiskForecast,
                                alerts: List[Alert], analogues: List[Analogue],
                                explanation: Explanation) -> plt.Figure
    def export_html_report(self, region: str, ...) -> Path
```

### Wave 4: Integration

#### End-to-end demo notebook

`notebooks/02_end_to_end_demo.ipynb` demonstrating:
1. Load harmonized data
2. Generate embeddings (TS + text)
3. Train baseline forecaster
4. Build similarity index
5. Run anomaly detection
6. Generate alerts
7. Create explainability report
8. Visualize dashboard

#### CLI updates

Update `scripts/run_extractive_risk_pipeline.py` with:
- `forecast` command using RiskForecaster
- `detect-anomalies` command using AnomalyDetector
- `find-analogues` command using SimilaritySearch
- `report` command generating full analysis

## Implementation Plan

| Wave | Modules | Parallelizable | Est. Lines |
|------|---------|----------------|------------|
| 1 | `baseline.py`, `forecaster.py` | Yes (2 agents) | ~350 |
| 2 | `similarity.py`, `detector.py`, `early_warning.py` | Yes (3 agents) | ~530 |
| 3 | `explainability.py`, `visualization.py` | Yes (2 agents) | ~450 |
| 4 | Integration notebook + CLI | Sequential | ~300 |

**Total:** ~1,630 lines across 7 modules + notebook

## Technical Decisions

1. **Vector DB:** FAISS (file-based) for MVP, designed for easy swap to pgvector
2. **Forecasting:** Prophet as default (better for irregular data), ARIMA as fallback
3. **Anomaly Detection:** Ensemble of 3 methods for robustness
4. **Visualization:** Matplotlib/Seaborn with consistent styling
5. **RAG:** Template-based narratives now, LLM integration point ready

## Success Criteria

- [ ] All imports in `__init__.py` files resolve without errors
- [ ] End-to-end notebook runs successfully
- [ ] CLI commands work for all operations
- [ ] Dashboard visualization renders for sample region
- [ ] Unit tests pass for each module
