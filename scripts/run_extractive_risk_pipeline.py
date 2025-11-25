#!/usr/bin/env python3
"""Main CLI pipeline for PI-ERE.

This script provides a command-line interface for running the end-to-end
Predictive Intelligence pipeline for extractive industries.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pi_ere.data.ingest import DataIngestionOrchestrator, get_default_regions
from pi_ere.data.sources import ACLEDSource, CommoditiesSource, GDELTSource, WorldBankSource
from pi_ere.data.harmonize import DataHarmonizer
from pi_ere.embeddings import TextEmbedder, TimeSeriesEmbedder
from pi_ere.utils.config import config


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


if __name__ == '__main__':
    cli()
