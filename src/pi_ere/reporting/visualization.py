"""Visualization tools for risk forecasts, alerts, and anomalies.

Provides charting and dashboard capabilities for the PI-ERE project,
including risk trajectories, feature importance, analogues, and alert timelines.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

if TYPE_CHECKING:
    from pi_ere.anomaly.detector import Anomaly
    from pi_ere.anomaly.early_warning import Alert
    from pi_ere.models.forecaster import RiskForecast

# Try to import UMAP, fall back to PCA
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logger.warning("UMAP not available, will use PCA for dimensionality reduction")

from sklearn.decomposition import PCA


# Color schemes
RISK_COLORS = {
    'low': '#2ecc71',      # green
    'moderate': '#f39c12',  # orange
    'high': '#e74c3c',     # red
}

ALERT_COLORS = {
    'watch': '#3498db',     # blue
    'warning': '#f39c12',   # orange
    'critical': '#e74c3c',  # red
}


class RiskVisualizer:
    """Visualization tools for risk forecasts and alerts.

    Provides methods for creating charts, dashboards, and HTML reports
    for risk assessment visualization.
    """

    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualizer with style settings.

        Args:
            style: Matplotlib style name (default: 'default')
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize

        # Apply style
        plt.style.use(style)
        sns.set_palette("husl")

        # Set style parameters
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })

        logger.info(f"Initialized RiskVisualizer (style={style}, figsize={figsize})")

    def plot_risk_trajectory(
        self,
        forecast: 'RiskForecast',
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Plot risk forecast trajectory with confidence bands.

        Args:
            forecast: RiskForecast object with predictions
            ax: Optional matplotlib axes to plot on

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        df = forecast.risk_scores

        # Plot mean trajectory
        ax.plot(df['date'], df['score'],
                color='#3498db', linewidth=2, label='Forecast')

        # Plot confidence bands
        ax.fill_between(
            df['date'],
            df['lower'],
            df['upper'],
            alpha=0.3,
            color='#3498db',
            label='Confidence Interval'
        )

        # Add horizontal lines for risk levels
        ax.axhline(y=0.3, color=RISK_COLORS['low'],
                   linestyle='--', alpha=0.5, label='Low Risk')
        ax.axhline(y=0.6, color=RISK_COLORS['moderate'],
                   linestyle='--', alpha=0.5, label='Moderate Risk')
        ax.axhline(y=0.8, color=RISK_COLORS['high'],
                   linestyle='--', alpha=0.5, label='High Risk')

        ax.set_xlabel('Date')
        ax.set_ylabel('Risk Score')
        ax.set_title(f'Risk Trajectory for {forecast.region}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        logger.debug(f"Plotted risk trajectory for {forecast.region}")

        return fig

    def plot_feature_importance(
        self,
        explanation: Dict[str, float],
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Plot feature importance as horizontal bar chart.

        Args:
            explanation: Dictionary mapping feature names to importance scores
            ax: Optional matplotlib axes to plot on

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Sort by importance
        sorted_items = sorted(explanation.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_items[:10]]  # Top 10
        importances = [item[1] for item in sorted_items[:10]]

        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))

        ax.barh(y_pos, importances, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')

        logger.debug(f"Plotted feature importance for {len(features)} features")

        return fig

    def plot_analogues(
        self,
        analogues: List[Dict[str, Any]],
        current_embedding: np.ndarray,
        all_embeddings: Optional[np.ndarray] = None,
        method: str = 'umap'
    ) -> plt.Figure:
        """Plot analogues in 2D space using dimensionality reduction.

        Args:
            analogues: List of analogue dictionaries with 'embedding' key
            current_embedding: Embedding vector for current region
            all_embeddings: Optional array of all embeddings for context
            method: Dimensionality reduction method ('umap' or 'pca')

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Collect embeddings
        analogue_embeddings = np.array([a['embedding'] for a in analogues])

        # Combine all embeddings for fitting reducer
        if all_embeddings is not None:
            fit_embeddings = all_embeddings
        else:
            fit_embeddings = np.vstack([analogue_embeddings, current_embedding.reshape(1, -1)])

        # Apply dimensionality reduction
        if method == 'umap' and HAS_UMAP:
            reducer = UMAP(n_components=2, random_state=42)
            logger.debug("Using UMAP for dimensionality reduction")
        else:
            reducer = PCA(n_components=2)
            logger.debug("Using PCA for dimensionality reduction")

        reduced_fit = reducer.fit_transform(fit_embeddings)

        # Transform analogues and current
        if all_embeddings is not None:
            # Find indices in all_embeddings
            reduced_analogues = reduced_fit[:len(analogue_embeddings)]
            reduced_current = reducer.transform(current_embedding.reshape(1, -1))
        else:
            reduced_analogues = reduced_fit[:-1]
            reduced_current = reduced_fit[-1].reshape(1, -1)

        # Plot analogues
        ax.scatter(
            reduced_analogues[:, 0],
            reduced_analogues[:, 1],
            c='#3498db',
            s=100,
            alpha=0.6,
            label='Analogues'
        )

        # Plot current region
        ax.scatter(
            reduced_current[:, 0],
            reduced_current[:, 1],
            c='#e74c3c',
            s=200,
            marker='*',
            edgecolors='black',
            linewidths=1.5,
            label='Current Region'
        )

        # Add labels for top analogues
        for i, analogue in enumerate(analogues[:5]):
            ax.annotate(
                analogue.get('region', f'A{i}'),
                (reduced_analogues[i, 0], reduced_analogues[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )

        ax.set_xlabel(f'Component 1')
        ax.set_ylabel(f'Component 2')
        ax.set_title(f'Historical Analogues ({method.upper()})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        logger.debug(f"Plotted {len(analogues)} analogues using {method}")

        return fig

    def plot_alert_timeline(
        self,
        alerts: List['Alert'],
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Plot timeline of alerts colored by severity level.

        Args:
            alerts: List of Alert objects
            ax: Optional matplotlib axes to plot on

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        if not alerts:
            ax.text(0.5, 0.5, 'No alerts',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Alert Timeline')
            return fig

        # Organize alerts by level
        by_level = {'watch': [], 'warning': [], 'critical': []}
        for alert in alerts:
            if alert.level in by_level:
                by_level[alert.level].append(alert)

        # Plot each level
        y_positions = {'watch': 1, 'warning': 2, 'critical': 3}

        for level, level_alerts in by_level.items():
            if level_alerts:
                timestamps = [a.timestamp for a in level_alerts]
                y_vals = [y_positions[level]] * len(timestamps)

                ax.scatter(
                    timestamps,
                    y_vals,
                    c=ALERT_COLORS[level],
                    s=100,
                    alpha=0.7,
                    label=level.capitalize()
                )

        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Watch', 'Warning', 'Critical'])
        ax.set_xlabel('Time')
        ax.set_title('Alert Timeline')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='x')

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        logger.debug(f"Plotted timeline with {len(alerts)} alerts")

        return fig

    def plot_anomaly_scores(
        self,
        anomalies: List['Anomaly'],
        ax: Optional[plt.Axes] = None
    ) -> plt.Figure:
        """Plot anomaly scores over time.

        Args:
            anomalies: List of Anomaly objects
            ax: Optional matplotlib axes to plot on

        Returns:
            Figure object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        if not anomalies:
            ax.text(0.5, 0.5, 'No anomalies detected',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Anomaly Scores')
            return fig

        # Extract data
        timestamps = [a.timestamp for a in anomalies]
        severities = [a.severity for a in anomalies]

        # Color by severity
        colors = [ALERT_COLORS['critical'] if s >= 0.8
                 else ALERT_COLORS['warning'] if s >= 0.5
                 else ALERT_COLORS['watch'] for s in severities]

        # Plot
        ax.scatter(timestamps, severities, c=colors, s=100, alpha=0.7)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Warning Threshold')
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Critical Threshold')

        ax.set_xlabel('Time')
        ax.set_ylabel('Severity Score')
        ax.set_title('Anomaly Scores')
        ax.set_ylim([0, 1])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        logger.debug(f"Plotted {len(anomalies)} anomaly scores")

        return fig

    def create_region_dashboard(
        self,
        region: str,
        forecast: Optional['RiskForecast'] = None,
        alerts: Optional[List['Alert']] = None,
        analogues: Optional[Dict[str, Any]] = None,
        explanation: Optional[Dict[str, float]] = None
    ) -> plt.Figure:
        """Create comprehensive dashboard for a region.

        Args:
            region: Region identifier
            forecast: Optional RiskForecast object
            alerts: Optional list of Alert objects
            analogues: Optional dict with 'analogues', 'current_embedding' keys
            explanation: Optional feature importance dictionary

        Returns:
            Figure with 2x2 subplot grid
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Risk Dashboard: {region}', fontsize=16, fontweight='bold')

        # Plot 1: Risk trajectory
        if forecast:
            self.plot_risk_trajectory(forecast, ax=axes[0, 0])
        else:
            axes[0, 0].text(0.5, 0.5, 'No forecast data',
                          ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Risk Trajectory')

        # Plot 2: Feature importance
        if explanation:
            self.plot_feature_importance(explanation, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'No explanation data',
                          ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Feature Importance')

        # Plot 3: Anomaly scores (or alert timeline if no anomalies)
        if alerts:
            # Extract anomalies from alerts
            anomalies = []
            for alert in alerts:
                if hasattr(alert, 'related_anomalies') and alert.related_anomalies:
                    anomalies.extend(alert.related_anomalies)

            if anomalies:
                self.plot_anomaly_scores(anomalies, ax=axes[1, 0])
            else:
                self.plot_alert_timeline(alerts, ax=axes[1, 0])
        else:
            axes[1, 0].text(0.5, 0.5, 'No anomaly data',
                          ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Anomalies')

        # Plot 4: Alert timeline
        if alerts:
            self.plot_alert_timeline(alerts, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'No alerts',
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Alerts')

        plt.tight_layout()

        logger.info(f"Created dashboard for region '{region}'")

        return fig

    def export_html_report(
        self,
        region: str,
        figures: Dict[str, plt.Figure],
        output_dir: Optional[Path] = None
    ) -> Path:
        """Export figures to HTML report.

        Args:
            region: Region identifier
            figures: Dictionary mapping names to Figure objects
            output_dir: Output directory (default: './reports')

        Returns:
            Path to generated HTML file
        """
        if output_dir is None:
            output_dir = Path('./reports')

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save figures as PNG
        image_paths = {}
        for name, fig in figures.items():
            img_path = output_dir / f'{region}_{name}.png'
            fig.savefig(img_path, dpi=150, bbox_inches='tight')
            image_paths[name] = img_path.name
            logger.debug(f"Saved figure '{name}' to {img_path}")

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Risk Report: {region}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .report-info {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .figure {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
        }}
        .figure h2 {{
            color: #34495e;
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <h1>Risk Assessment Report: {region}</h1>
    <div class="report-info">
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Region:</strong> {region}</p>
    </div>
"""

        # Add figures
        for name, img_name in image_paths.items():
            html_content += f"""
    <div class="figure">
        <h2>{name.replace('_', ' ').title()}</h2>
        <img src="{img_name}" alt="{name}">
    </div>
"""

        html_content += """
</body>
</html>
"""

        # Write HTML file
        html_path = output_dir / f'{region}_report.html'
        html_path.write_text(html_content)

        logger.info(f"Exported HTML report to {html_path}")

        return html_path
