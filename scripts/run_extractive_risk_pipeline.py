#!/usr/bin/env python3
"""Main CLI pipeline for PI-ERE.

This script provides a command-line interface for running the end-to-end
Predictive Intelligence pipeline for extractive industries.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
import matplotlib.pyplot as plt
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_ere.data.ingest import DataIngestionOrchestrator, get_default_regions
from pi_ere.data.sources import ACLEDSource, CommoditiesSource, GDELTSource, WorldBankSource
from pi_ere.data.harmonize import DataHarmonizer
from pi_ere.embeddings import TextEmbedder, TimeSeriesEmbedder
from pi_ere.utils.config import config
from pi_ere.models import RiskForecaster, RiskForecast
from pi_ere.vector_db import SimilaritySearch
from pi_ere.anomaly import AnomalyDetector, EarlyWarningSystem
from pi_ere.reporting import ExplainabilityEngine, RiskVisualizer


# Configure logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """PI-ERE: Predictive Intelligence for Energy, Resources & Extractives.

    A predictive intelligence system for operational risk forecasting in
    extractive industries operating in fragile regions.
    """
    pass


@cli.command()
@click.option(
    '--regions',
    '-r',
    multiple=True,
    help='ISO country codes to fetch data for (e.g., COD, MLI). If not specified, uses default regions.'
)
@click.option(
    '--start-date',
    '-s',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
    help='Start date for data ingestion (YYYY-MM-DD)'
)
@click.option(
    '--end-date',
    '-e',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default=datetime.now().strftime("%Y-%m-%d"),
    help='End date for data ingestion (YYYY-MM-DD)'
)
@click.option(
    '--sources',
    multiple=True,
    type=click.Choice(['acled', 'gdelt', 'world_bank', 'commodities'], case_sensitive=False),
    help='Data sources to ingest. If not specified, ingests from all enabled sources.'
)
def ingest(regions, start_date, end_date, sources):
    """Ingest data from OSINT sources.

    Examples:
        # Ingest all sources for default regions (last year)
        python run_extractive_risk_pipeline.py ingest

        # Ingest specific sources for specific regions
        python run_extractive_risk_pipeline.py ingest -r COD -r MLI -s acled -s gdelt --start-date 2023-01-01
    """
    logger.info("=" * 80)
    logger.info("PI-ERE DATA INGESTION")
    logger.info("=" * 80)

    # Setup regions
    region_list = list(regions) if regions else get_default_regions()
    logger.info(f"Regions: {', '.join(region_list)}")

    # Setup orchestrator
    orchestrator = DataIngestionOrchestrator()

    # Register sources
    logger.info("Registering data sources...")

    if not sources or 'acled' in sources:
        orchestrator.register_source(ACLEDSource())

    if not sources or 'gdelt' in sources:
        orchestrator.register_source(GDELTSource())

    if not sources or 'world_bank' in sources:
        orchestrator.register_source(WorldBankSource())

    if not sources or 'commodities' in sources:
        orchestrator.register_source(CommoditiesSource())

    # Ingest data
    logger.info(f"Ingesting data from {start_date.date()} to {end_date.date()}...")

    results = orchestrator.ingest_all(
        start_date=start_date,
        end_date=end_date,
        regions=region_list,
        sources=list(sources) if sources else None,
        save_raw=True,
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 80)

    for source_name, df in results.items():
        logger.info(f"{source_name}: {len(df)} records")

    logger.info("\nIngestion complete!")


@cli.command()
@click.option(
    '--output-file',
    '-o',
    type=click.Path(),
    default='harmonized_data.parquet',
    help='Output file for harmonized data'
)
@click.option(
    '--granularity',
    '-g',
    type=click.Choice(['D', 'W', 'M']),
    default='D',
    help='Time granularity (D=daily, W=weekly, M=monthly)'
)
def harmonize(output_file, granularity):
    """Harmonize ingested data into unified time-series panel.

    This combines data from multiple sources into a standardized format
    suitable for embedding and modeling.

    Examples:
        # Harmonize with daily granularity
        python run_extractive_risk_pipeline.py harmonize

        # Harmonize with weekly granularity
        python run_extractive_risk_pipeline.py harmonize -g W
    """
    logger.info("=" * 80)
    logger.info("PI-ERE DATA HARMONIZATION")
    logger.info("=" * 80)

    # Load latest raw data from each source
    # (In production, this would load from specific files)
    orchestrator = DataIngestionOrchestrator()

    # Register sources to get their latest files
    sources = [ACLEDSource(), GDELTSource(), WorldBankSource(), CommoditiesSource()]

    data_dict = {}

    for source in sources:
        if not source.enabled:
            continue

        latest_file = source.get_latest_file()

        if latest_file:
            logger.info(f"Loading {source.name} from {latest_file.name}")
            df = source.load_raw(latest_file.name)
            data_dict[source.name] = df
        else:
            logger.warning(f"No data files found for {source.name}")

    if not data_dict:
        logger.error("No data available to harmonize. Run 'ingest' first.")
        return

    # Harmonize
    harmonizer = DataHarmonizer(granularity=granularity)

    logger.info("Harmonizing data...")
    harmonized = harmonizer.harmonize(data_dict)

    # Create features
    logger.info("Engineering features...")
    harmonized = harmonizer.create_features(harmonized)

    # Save
    output_path = harmonizer.save_harmonized(harmonized, filename=output_file)

    logger.info(f"\nHarmonized data saved to: {output_path}")
    logger.info(f"Shape: {harmonized.shape}")
    logger.info(f"Regions: {harmonized['region'].nunique()}")
    logger.info(f"Features: {harmonized['feature_name'].nunique()}")
    logger.info(f"Date range: {harmonized['date'].min()} to {harmonized['date'].max()}")


@cli.command()
@click.option(
    '--input-file',
    '-i',
    type=click.Path(exists=True),
    required=True,
    help='Harmonized data file'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='data/embeddings',
    help='Output directory for embeddings'
)
@click.option(
    '--model-type',
    '-m',
    type=click.Choice(['ts2vec', 'text']),
    required=True,
    help='Type of embedding model to use'
)
def embed(input_file, output_dir, model_type):
    """Generate embeddings from harmonized data.

    Examples:
        # Generate time-series embeddings
        python run_extractive_risk_pipeline.py embed -i harmonized_data.parquet -m ts2vec

        # Generate text embeddings
        python run_extractive_risk_pipeline.py embed -i harmonized_data.parquet -m text
    """
    logger.info("=" * 80)
    logger.info("PI-ERE EMBEDDING GENERATION")
    logger.info("=" * 80)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load harmonized data
    import pandas as pd
    logger.info(f"Loading data from {input_file}")
    df = pd.read_parquet(input_file)

    if model_type == 'ts2vec':
        logger.info("Generating time-series embeddings...")

        # Convert to wide format for time-series embedding
        wide_df = df.pivot_table(
            index=['region', 'date'],
            columns='feature_name',
            values='value'
        ).reset_index()

        # Prepare data for embedding (region-wise time series)
        # This is a simplified version - full implementation would handle this more carefully
        logger.info("Time-series embedding not yet fully implemented in CLI")
        logger.info("Use notebooks for detailed embedding workflow")

    elif model_type == 'text':
        logger.info("Text embedding requires text data (news, events)")
        logger.info("Use notebooks for text embedding workflow")

    logger.info("\nEmbedding complete!")


@cli.command()
def status():
    """Show status of data sources and pipeline.

    Displays information about registered data sources, latest data files,
    and system configuration.
    """
    logger.info("=" * 80)
    logger.info("PI-ERE SYSTEM STATUS")
    logger.info("=" * 80)

    # Setup orchestrator
    orchestrator = DataIngestionOrchestrator()

    # Register all sources
    sources = [ACLEDSource(), GDELTSource(), WorldBankSource(), CommoditiesSource()]

    for source in sources:
        orchestrator.register_source(source)

    # Get status
    status_df = orchestrator.get_ingestion_status()

    logger.info("\nData Sources:")
    logger.info("-" * 80)

    for _, row in status_df.iterrows():
        status_icon = "✓" if row['enabled'] else "✗"
        logger.info(f"\n{status_icon} {row['source'].upper()}")
        logger.info(f"  Enabled: {row['enabled']}")
        logger.info(f"  Update Frequency: {row['update_frequency']}")
        logger.info(f"  Latest File: {row['latest_file'] or 'None'}")
        logger.info(f"  Latest Date: {row['latest_file_date'] or 'N/A'}")

    # Configuration
    logger.info("\n" + "=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Data Directory: {config.data_dir}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Embedding Model: {config.get('embedding.ts_embedding.model_type')}")


@cli.command()
@click.option(
    '--days',
    '-d',
    type=int,
    default=7,
    help='Number of days to look back'
)
def update(days):
    """Perform incremental data update.

    Fetches recent data from all enabled sources.

    Examples:
        # Update with last 7 days of data
        python run_extractive_risk_pipeline.py update

        # Update with last 30 days
        python run_extractive_risk_pipeline.py update -d 30
    """
    logger.info("=" * 80)
    logger.info(f"PI-ERE INCREMENTAL UPDATE ({days} days)")
    logger.info("=" * 80)

    orchestrator = DataIngestionOrchestrator()

    # Register sources
    orchestrator.register_source(ACLEDSource())
    orchestrator.register_source(GDELTSource())
    orchestrator.register_source(WorldBankSource())
    orchestrator.register_source(CommoditiesSource())

    # Incremental ingest
    results = orchestrator.ingest_incremental(
        lookback_days=days,
        regions=get_default_regions(),
    )

    # Summary
    logger.info("\nUpdate complete!")

    for source_name, df in results.items():
        logger.info(f"{source_name}: {len(df)} new records")


@cli.command()
@click.option(
    '--region',
    '-r',
    required=True,
    help='Region code to forecast (e.g., COD, MLI)'
)
@click.option(
    '--input-file',
    '-i',
    type=click.Path(exists=True),
    required=True,
    help='Harmonized data file'
)
@click.option(
    '--horizon',
    '-h',
    type=int,
    default=6,
    help='Forecast horizon in months'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output file for forecast results'
)
def forecast(region, input_file, horizon, output):
    """Generate risk forecasts for a specific region.

    Examples:
        # Forecast 6 months ahead for DRC
        python run_extractive_risk_pipeline.py forecast -r COD -i harmonized_data.parquet

        # Forecast 12 months with custom output
        python run_extractive_risk_pipeline.py forecast -r MLI -i harmonized_data.parquet -h 12 -o forecast_mali.json
    """
    logger.info("=" * 80)
    logger.info("PI-ERE RISK FORECASTING")
    logger.info("=" * 80)

    try:
        # Load harmonized data
        import pandas as pd
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)

        # Filter for target region
        region_data = df[df['region'] == region]
        if region_data.empty:
            logger.error(f"No data found for region: {region}")
            return

        logger.info(f"Region: {region}")
        logger.info(f"Forecast horizon: {horizon} months")
        logger.info(f"Data records: {len(region_data)}")

        # Initialize forecaster
        logger.info("Initializing risk forecaster...")
        forecaster = RiskForecaster()

        # Generate forecast
        logger.info(f"Generating {horizon}-month forecast...")
        forecast_result = forecaster.forecast(
            data=region_data,
            region=region,
            horizon=horizon
        )

        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving forecast to {output}")
            forecast_result.save(output_path)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("FORECAST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Region: {region}")
        logger.info(f"Forecast horizon: {horizon} months")
        logger.info(f"Mean risk score: {forecast_result.mean_risk:.3f}")
        logger.info(f"Risk trend: {forecast_result.trend}")
        logger.info("\nForecasting complete!")

    except Exception as e:
        logger.error(f"Error during forecasting: {e}")
        raise


@cli.command()
@click.option(
    '--input-file',
    '-i',
    type=click.Path(exists=True),
    required=True,
    help='Harmonized data file'
)
@click.option(
    '--region',
    '-r',
    help='Region to analyze (optional, all regions if not specified)'
)
@click.option(
    '--methods',
    default='residual,isolation_forest',
    help='Comma-separated detection methods (default: residual,isolation_forest)'
)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    help='Output file for anomaly results'
)
def detect_anomalies(input_file, region, methods, output):
    """Run anomaly detection on harmonized data.

    Examples:
        # Detect anomalies for all regions
        python run_extractive_risk_pipeline.py detect-anomalies -i harmonized_data.parquet

        # Detect for specific region
        python run_extractive_risk_pipeline.py detect-anomalies -i harmonized_data.parquet -r COD

        # Custom methods and output
        python run_extractive_risk_pipeline.py detect-anomalies -i harmonized_data.parquet --methods residual,isolation_forest,lof -o anomalies.json
    """
    logger.info("=" * 80)
    logger.info("PI-ERE ANOMALY DETECTION")
    logger.info("=" * 80)

    try:
        # Load harmonized data
        import pandas as pd
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)

        # Filter for region if specified
        if region:
            df = df[df['region'] == region]
            if df.empty:
                logger.error(f"No data found for region: {region}")
                return
            logger.info(f"Region: {region}")
        else:
            logger.info(f"Analyzing all regions: {df['region'].nunique()} regions")

        # Parse methods
        method_list = [m.strip() for m in methods.split(',')]
        logger.info(f"Detection methods: {', '.join(method_list)}")

        # Initialize detector
        logger.info("Initializing anomaly detector...")
        detector = AnomalyDetector(methods=method_list)

        # Detect anomalies
        logger.info("Running anomaly detection...")
        anomaly_list = detector.detect(df)

        # Convert list of Anomaly objects to DataFrame
        anomalies = pd.DataFrame([
            {
                'timestamp': a.timestamp,
                'anomaly_type': a.anomaly_type,
                'feature': a.feature,
                'severity': a.severity,
                'description': a.description,
                'raw_score': a.raw_score
            }
            for a in anomaly_list
        ])

        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving anomalies to {output}")
            anomalies.to_json(output_path, orient='records', indent=2)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("ANOMALY DETECTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total anomalies detected: {len(anomalies)}")

        if len(anomalies) > 0:
            logger.info(f"Anomaly types: {anomalies['anomaly_type'].value_counts().to_dict()}")

            # Top anomalies by severity
            if 'severity' in anomalies.columns:
                logger.info("\nTop 5 anomalies by severity:")
                top_anomalies = anomalies.nlargest(5, 'severity')
                for _, row in top_anomalies.iterrows():
                    logger.info(f"  {row['anomaly_type']}: {row['feature']} (severity: {row['severity']:.3f})")

        logger.info("\nAnomaly detection complete!")

    except Exception as e:
        logger.error(f"Error during anomaly detection: {e}")
        raise


@cli.command()
@click.option(
    '--region',
    '-r',
    required=True,
    help='Target region to find analogues for'
)
@click.option(
    '--embeddings-file',
    '-e',
    type=click.Path(exists=True),
    required=True,
    help='Embeddings file'
)
@click.option(
    '--top-k',
    '-k',
    type=int,
    default=10,
    help='Number of analogues to return'
)
def find_analogues(region, embeddings_file, top_k):
    """Find historical analogues for a region.

    Uses similarity search over embeddings to identify historical periods
    or other regions with similar risk profiles.

    Examples:
        # Find top 10 analogues for DRC
        python run_extractive_risk_pipeline.py find-analogues -r COD -e embeddings.npz

        # Find top 20 analogues
        python run_extractive_risk_pipeline.py find-analogues -r MLI -e embeddings.npz -k 20
    """
    logger.info("=" * 80)
    logger.info("PI-ERE ANALOGUE SEARCH")
    logger.info("=" * 80)

    try:
        logger.info(f"Target region: {region}")
        logger.info(f"Embeddings file: {embeddings_file}")
        logger.info(f"Number of analogues: {top_k}")

        # Load embeddings and metadata
        logger.info("Loading embeddings...")
        embeddings_path = Path(embeddings_file)

        if embeddings_path.suffix == '.npz':
            data = np.load(embeddings_path, allow_pickle=True)
            embeddings = data['embeddings']
            metadata = pd.DataFrame(data['metadata'].item()) if 'metadata' in data else None
        elif embeddings_path.suffix == '.npy':
            embeddings = np.load(embeddings_path)
            # Try to load metadata from companion file
            metadata_path = embeddings_path.parent / 'ts_metadata.parquet'
            metadata = pd.read_parquet(metadata_path) if metadata_path.exists() else None
        else:
            raise ValueError(f"Unsupported embeddings format: {embeddings_path.suffix}")

        # Initialize similarity search
        logger.info("Initializing similarity search index...")
        embedding_dim = embeddings.shape[1]
        similarity_search = SimilaritySearch(embedding_dim=embedding_dim)

        # Add embeddings to index
        if metadata is not None:
            ids = [f"{r}_{d}" for r, d in zip(metadata['region'], metadata['date'])]
            similarity_search.add_embeddings(ids, embeddings, metadata)
            logger.info(f"Added {len(embeddings)} embeddings to index")

        # Get query embedding (latest for target region)
        if metadata is not None:
            region_mask = metadata['region'] == region
            if not region_mask.any():
                logger.error(f"No embeddings found for region: {region}")
                return
            region_embeddings = embeddings[region_mask]
            query_embedding = region_embeddings[-1]  # Latest
        else:
            logger.error("Cannot search without metadata")
            return

        # Find analogues
        logger.info(f"Searching for top {top_k} analogues...")
        analogues = similarity_search.find_analogues(
            region=region,
            current_embedding=query_embedding,
            exclude_recent_days=90
        )

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("ANALOGUE SEARCH RESULTS")
        logger.info("=" * 80)
        logger.info(f"Target region: {region}")
        logger.info(f"\nTop {len(analogues)} analogues:")

        for i, analogue in enumerate(analogues[:top_k], 1):
            logger.info(f"\n{i}. {analogue.region} ({analogue.start_date.strftime('%Y-%m-%d')})")
            logger.info(f"   Similarity: {analogue.similarity_score:.3f}")
            logger.info(f"   Outcome: {analogue.outcome}")

        logger.info("\nAnalogue search complete!")

    except Exception as e:
        logger.error(f"Error during analogue search: {e}")
        raise


@cli.command()
@click.option(
    '--region',
    '-r',
    required=True,
    help='Region code for report'
)
@click.option(
    '--input-file',
    '-i',
    type=click.Path(exists=True),
    required=True,
    help='Harmonized data file'
)
@click.option(
    '--output-dir',
    '-o',
    type=click.Path(),
    default='reports/',
    help='Output directory for report'
)
def report(region, input_file, output_dir):
    """Generate full risk report for a region.

    Orchestrates the complete pipeline: forecasting, anomaly detection,
    early warning signals, and explanation generation. Produces an HTML report.

    Examples:
        # Generate report for DRC
        python run_extractive_risk_pipeline.py report -r COD -i harmonized_data.parquet

        # Generate report with custom output directory
        python run_extractive_risk_pipeline.py report -r MLI -i harmonized_data.parquet -o custom_reports/
    """
    logger.info("=" * 80)
    logger.info("PI-ERE RISK REPORT GENERATION")
    logger.info("=" * 80)

    try:
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load harmonized data
        import pandas as pd
        logger.info(f"Loading data from {input_file}")
        df = pd.read_parquet(input_file)

        # Filter for target region
        region_data = df[df['region'] == region]
        if region_data.empty:
            logger.error(f"No data found for region: {region}")
            return

        logger.info(f"Region: {region}")
        logger.info(f"Data records: {len(region_data)}")

        # Step 1: Generate forecast
        logger.info("\n" + "-" * 80)
        logger.info("Step 1/5: Generating risk forecast...")
        logger.info("-" * 80)
        forecaster = RiskForecaster()
        forecaster.fit(df, region)  # Fit on full harmonized data
        forecast_result = forecaster.predict_risk(region, horizon_months=6)
        mean_risk = forecast_result.risk_scores['score'].mean()
        logger.info(f"Forecast generated: Mean risk = {mean_risk:.3f}")

        # Step 2: Run anomaly detection
        logger.info("\n" + "-" * 80)
        logger.info("Step 2/5: Running anomaly detection...")
        logger.info("-" * 80)
        detector = AnomalyDetector(methods=['residual', 'isolation_forest'])
        anomalies = detector.detect(region_data)
        logger.info(f"Anomalies detected: {len(anomalies)}")

        # Step 3: Generate early warning alerts
        logger.info("\n" + "-" * 80)
        logger.info("Step 3/5: Generating early warning alerts...")
        logger.info("-" * 80)
        early_warning = EarlyWarningSystem()
        alerts = early_warning.analyze(
            region=region,
            data=region_data,
            forecast=forecast_result
        )
        logger.info(f"Alerts generated: {len(alerts)}")

        # Step 4: Create explanations
        logger.info("\n" + "-" * 80)
        logger.info("Step 4/5: Creating explanations...")
        logger.info("-" * 80)
        explainer = ExplainabilityEngine()
        explanation = explainer.explain_forecast(
            forecast=forecast_result,
            data=region_data
        )
        logger.info("Explanations created")

        # Step 5: Export HTML report
        logger.info("\n" + "-" * 80)
        logger.info("Step 5/5: Exporting HTML report...")
        logger.info("-" * 80)
        visualizer = RiskVisualizer()

        # Create dashboard figure
        feature_importance = explanation.feature_importance if hasattr(explanation, 'feature_importance') else None
        dashboard_fig = visualizer.create_region_dashboard(
            region=region,
            forecast=forecast_result,
            alerts=alerts,
            explanation=feature_importance
        )

        # Export to HTML
        report_file = visualizer.export_html_report(
            region=region,
            figures={'dashboard': dashboard_fig},
            output_dir=output_path
        )
        plt.close(dashboard_fig)  # Clean up
        logger.info(f"Report saved to: {report_file}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("REPORT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Region: {region}")
        logger.info(f"Forecast horizon: 6 months")
        logger.info(f"Mean risk score: {mean_risk:.3f}")
        logger.info(f"Anomalies detected: {len(anomalies)}")
        logger.info(f"Alerts generated: {len(alerts)}")
        logger.info(f"\nReport location: {report_file}")
        logger.info("\nReport generation complete!")

    except Exception as e:
        logger.error(f"Error during report generation: {e}")
        raise


if __name__ == '__main__':
    cli()
