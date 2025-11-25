# Predictive Intelligence for Energy, Resources & Extractives (PI-ERE)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PI-ERE is a predictive intelligence module for energy, resources, and extractive industry operations in fragile or high-risk regions. It ingests multi-source open-source intelligence (OSINT) and transforms it into actionable, forward-looking risk assessments using state-of-the-art time-series embeddings and transformer models.

### Key Capabilities

- **3-6 Month Risk Forecasting**: Predict operational disruption probabilities using temporal transformer models
- **Early Warning Alerts**: Detect escalating tensions (labor unrest, community conflict, regulatory pressure) before they peak
- **Supply-Chain Anomaly Detection**: Identify unusual patterns that may indicate sanctions evasion, theft, or coercion
- **Analogue Search**: Find historical episodes similar to current conditions for contextual intelligence
- **Explainable AI**: Transparent attribution showing *why* risks are elevated and *what* drives predictions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│  (ACLED, GDELT, World Bank, Commodities, News)             │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              Data Harmonization & Preprocessing              │
│  (Time alignment, normalization, panel construction)        │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐   ┌──────────▼──────────┐
│  TS Embeddings   │   │  Text Embeddings    │
│   (TS2Vec)       │   │  (Sentence-BERT)    │
└───────┬──────────┘   └──────────┬──────────┘
        │                         │
        └────────────┬────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│          Temporal Transformer Forecaster                     │
│     (Chronos/TimesFM/TFT for risk trajectory)               │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────────┐   ┌──────────▼──────────┐
│ Anomaly Detection│   │  Vector Database     │
│ & Early Warning  │   │  (Similarity Search) │
└───────┬──────────┘   └──────────┬──────────┘
        │                         │
        └────────────┬────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│         Reporting & Explainability Layer                     │
│  (Risk scores, alerts, analogues, feature attribution)      │
└─────────────────────────────────────────────────────────────┘
```

## Use Cases

### 1. Asset-Level Operational Risk Forecast
Quantify the probability of operational disruption (protests, violence, permit suspension) for specific mining sites, pipelines, or terminals over 3-6 month horizons.

### 2. Community & Labor Unrest Early Warning
Detect escalating social tensions that could lead to blockades or strikes, with 4-12 week advance notice based on learned escalation patterns.

### 3. Supply-Chain Sanctions & Diversion Detection
Identify anomalous transaction flows or counterparty relationships that resemble known evasion or coercion patterns.

## Project Structure

```
pi-ere/
├── README.md                       # This file
├── pyproject.toml                  # Python project configuration
├── config/                         # Configuration files
│   ├── data_sources.yaml          # Data source definitions
│   └── model_config.yaml          # Model hyperparameters
├── src/pi_ere/                    # Main package
│   ├── data/                      # Data ingestion & harmonization
│   │   ├── ingest.py             # Main ingestion orchestrator
│   │   ├── harmonize.py          # Data normalization & alignment
│   │   └── sources/              # Source-specific connectors
│   │       ├── acled.py          # Conflict & protest data
│   │       ├── gdelt.py          # News & events
│   │       ├── worldbank.py      # Macro indicators
│   │       └── commodities.py    # Commodity prices
│   ├── embeddings/               # Embedding models
│   │   ├── ts_embedder.py       # Time-series embedding (TS2Vec)
│   │   └── text_embedder.py     # Text embedding (SBERT)
│   ├── models/                   # Forecasting models
│   │   ├── baseline.py          # ARIMA/Prophet baselines
│   │   ├── transformer.py       # Temporal transformer
│   │   └── forecaster.py        # Main forecasting interface
│   ├── anomaly/                  # Anomaly detection
│   │   ├── detector.py          # Anomaly detection logic
│   │   └── early_warning.py     # Early warning system
│   ├── vector_db/                # Vector database & search
│   │   └── similarity.py        # Similarity search & analogues
│   ├── reporting/                # Output generation
│   │   ├── explainability.py    # Feature attribution & SHAP
│   │   └── visualization.py     # Plotting & dashboards
│   └── utils/                    # Utilities
│       └── config.py            # Configuration management
├── scripts/                       # CLI scripts
│   ├── run_extractive_risk_pipeline.py  # Main pipeline
│   └── setup_environment.sh      # Environment setup
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_forecasting.ipynb
│   ├── 03_embeddings_similarity.ipynb
│   └── 04_early_warning_alerts.ipynb
├── tests/                         # Unit tests
└── data/                          # Data storage
    ├── raw/                       # Raw ingested data
    ├── processed/                 # Harmonized data
    └── embeddings/                # Embedding vectors
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or poetry
- (Optional) CUDA-capable GPU for faster embedding/transformer training

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-org/pi-ere.git
cd pi-ere
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -e .
# OR
poetry install
```

4. **Configure data sources**
Edit `config/data_sources.yaml` with your API keys and data paths.

5. **Run setup script**
```bash
bash scripts/setup_environment.sh
```

## Quick Start

### 1. Data Ingestion
```bash
python scripts/run_extractive_risk_pipeline.py ingest \
    --regions "DRC,Mali,Afghanistan" \
    --start-date 2019-01-01 \
    --end-date 2024-12-31
```

### 2. Train Embedding Models
```bash
python scripts/run_extractive_risk_pipeline.py embed \
    --model-type ts2vec \
    --output-dir data/embeddings/
```

### 3. Generate Risk Forecasts
```bash
python scripts/run_extractive_risk_pipeline.py forecast \
    --region DRC \
    --asset "Tenke Fungurume Mine" \
    --horizon 6  # months
```

### 4. Run Anomaly Detection
```bash
python scripts/run_extractive_risk_pipeline.py detect-anomalies \
    --threshold 0.85 \
    --output reports/anomalies.json
```

### 5. Explore in Notebooks
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Data Sources (OSINT)

| Source | Type | Coverage | Update Frequency |
|--------|------|----------|------------------|
| **ACLED** | Conflict & protest events | Global (focus on fragile states) | Weekly |
| **GDELT** | News events & sentiment | Global | Daily |
| **World Bank** | Macro indicators, governance | 200+ countries | Quarterly/Annual |
| **FRED** | Commodity prices, FX rates | Global | Daily |
| **UCDP** | Armed conflict data | Global | Annual |

*Note: For production deployments, clients can integrate proprietary data (internal incident logs, supply-chain transactions, satellite imagery features).*

## Model Components

### Embedding Layer
- **TS2Vec**: Self-supervised contrastive learning for time-series representations
- **Sentence-BERT**: Text embeddings for news headlines and event descriptions
- **Fusion**: Multi-modal embedding concatenation or cross-attention

### Temporal Transformer
- **Architecture**: Chronos-inspired temporal transformer or Temporal Fusion Transformer (TFT)
- **Inputs**: Sequences of embeddings + structured features (counts, prices, indices)
- **Outputs**: Risk score trajectories with prediction intervals

### Anomaly Detection
- **Methods**:
  - Forecast residual-based (MAE/MSE thresholds)
  - Embedding drift (cosine distance from baseline)
  - Isolation Forest on multi-feature space
- **Alerts**: Composite risk score combining multiple signals

### Vector Database
- **Backend**: Qdrant or pgvector
- **Use Cases**:
  - Find past time periods similar to current conditions
  - Identify other regions exhibiting similar patterns
  - Supply-chain flow similarity matching

## Success Metrics (Phase 1)

| Metric | Target | Current |
|--------|--------|---------|
| Precision/Recall improvement vs manual baseline | ≥15-25% | TBD |
| False positive reduction (analyst eval) | ≥20% | TBD |
| Early warning lead time (backtest) | ≥4 weeks | TBD |
| Pipeline runtime (100 regions, 5 years) | <1 hour | TBD |

## Roadmap

### Phase 0: Skeleton & Data (Weeks 1-3) ✅
- [x] Repo structure
- [x] Basic data ingestion (ACLED, World Bank, commodities)
- [ ] Initial time-series visualization

### Phase 1: Baseline + Embeddings (Weeks 4-7)
- [ ] ARIMA/Prophet baseline forecasts
- [ ] TS2Vec embedding model
- [ ] Vector similarity search
- [ ] Embedding visualization (t-SNE/UMAP)

### Phase 2: Transformer Forecasting + Anomaly Layer (Weeks 8-13)
- [ ] Temporal transformer integration (Chronos/TimesFM)
- [ ] Risk scoring & prediction intervals
- [ ] Anomaly detection logic
- [ ] First end-to-end case study (DRC or Mali)

### Phase 3: Extractives Story & Polish (Weeks 14-16)
- [ ] 2-3 realistic regional narratives
- [ ] Analogue search examples
- [ ] Explainability dashboard
- [ ] Documentation & PRD alignment

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use PI-ERE in your research or operations, please cite:

```bibtex
@software{pi_ere_2025,
  title={PI-ERE: Predictive Intelligence for Energy, Resources & Extractives},
  author={Your Organization},
  year={2025},
  url={https://github.com/your-org/pi-ere}
}
```

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/your-org/pi-ere/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/pi-ere/discussions)
- **Email**: pi-ere-support@your-org.com

## Acknowledgments

This project builds on cutting-edge research in:
- Time-series foundation models (Chronos, TimesFM)
- Self-supervised time-series learning (TS2Vec, CoST)
- Temporal transformers (TFT, Informer, Autoformer)
- Geopolitical risk forecasting methodologies

---

**Disclaimer**: PI-ERE is designed for operational risk assessment and analyst augmentation. It is not a replacement for human judgment, local expertise, or ethical due diligence. Users are responsible for validating model outputs and ensuring compliance with applicable laws and regulations.
