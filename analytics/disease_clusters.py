"""Spatial Clustering & Temporal Trend Analytics for PashuRaksha AI.

Applies DBSCAN (Haversine metric) to identify emerging geospatial disease hotspots
and rolling temporal case aggregations when real coordinates and dates exist.
Outputs:
  - reports/spatial_disease_clusters.png
  - reports/temporal_trends.png
  - reports/cluster_summary.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.logger import get_logger

logger = get_logger("DiseaseClusters")


def analyze_disease_clusters(
    clinical_file: str = "data/clinical/clinical_data.csv",
    output_dir: str = "reports",
    eps_km: float = 35.0,
    min_samples: int = 4,
) -> Dict[str, Any]:
    """Detects spatial disease clusters and analyzes temporal case trajectories."""
    p = Path(clinical_file)
    if not p.exists():
        logger.warning(f"Clinical file not found: {clinical_file}")
        return {"error": "File not found"}

    df = pd.read_csv(p)
    required_cols = {"latitude", "longitude", "disease"}
    if not required_cols.issubset(df.columns):
        logger.warning(f"Dataset missing required spatial columns {required_cols}. Spatial analysis skipped.")
        return {"status": "skipped", "reason": "Missing spatial coordinates"}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Spatial DBSCAN (Haversine distance in radians)
    # -------------------------------------------------------------
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian

    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric="haversine")
    df["cluster_id"] = db.fit_predict(coords_rad)

    n_clusters = len(set(df["cluster_id"])) - (1 if -1 in df["cluster_id"] else 0)
    n_noise = int(np.sum(df["cluster_id"] == -1))

    logger.info(f"DBSCAN detected {n_clusters} potential emerging clusters ({n_noise} dispersed cases).")

    # Spatial Plot
    plt.figure(figsize=(9, 6))
    unique_clusters = sorted(df["cluster_id"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_clusters))))

    for cid, col in zip(unique_clusters, colors):
        subset = df[df["cluster_id"] == cid]
        if cid == -1:
            plt.scatter(
                subset["longitude"],
                subset["latitude"],
                c="gray",
                alpha=0.4,
                s=25,
                label="Dispersed Cases (Noise)",
            )
        else:
            plt.scatter(
                subset["longitude"],
                subset["latitude"],
                color=col,
                alpha=0.85,
                s=70,
                edgecolor="black",
                label=f"Potential Cluster #{cid} (n={len(subset)})",
            )

    plt.title("Livestock Disease Surveillance: Detected Spatial Clusters (DBSCAN)", fontsize=12, fontweight="bold")
    plt.xlabel("Longitude (°E)", fontweight="semibold")
    plt.ylabel("Latitude (°N)", fontweight="semibold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    spatial_plot_path = Path(output_dir) / "spatial_disease_clusters.png"
    plt.savefig(spatial_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 2. Temporal Trend Analysis
    # -------------------------------------------------------------
    temporal_summary = {}
    if "recorded_date" in df.columns:
        df["recorded_date"] = pd.to_datetime(df["recorded_date"])
        daily_counts = df.groupby(["recorded_date", "disease"]).size().unstack(fill_value=0)
        
        plt.figure(figsize=(10, 5))
        for col in daily_counts.columns:
            rolling_series = daily_counts[col].rolling(window=3, min_periods=1).mean()
            plt.plot(daily_counts.index, rolling_series, marker="o", label=col, linewidth=2)

        plt.title("Livestock Disease Surveillance: 3-Day Rolling Case Trajectory", fontsize=12, fontweight="bold")
        plt.xlabel("Date", fontweight="semibold")
        plt.ylabel("Reported Screening Cases", fontweight="semibold")
        plt.xticks(rotation=30)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        temporal_plot_path = Path(output_dir) / "temporal_trends.png"
        plt.savefig(temporal_plot_path, dpi=300)
        plt.close()

        temporal_summary = {
            "start_date": str(df["recorded_date"].min().date()),
            "end_date": str(df["recorded_date"].max().date()),
            "total_monitoring_days": int((df["recorded_date"].max() - df["recorded_date"].min()).days) + 1,
        }

    results = {
        "spatial_clustering": {
            "algorithm": "DBSCAN (Haversine)",
            "epsilon_km": eps_km,
            "min_samples": min_samples,
            "detected_clusters_count": n_clusters,
            "noise_cases_count": n_noise,
        },
        "temporal_analysis": temporal_summary,
        "epidemiological_notice": (
            "Detected spatial hotspots represent localized screening density anomalies. "
            "They do not constitute automated epidemic predictions without confirmatory diagnostic sampling."
        ),
    }

    report_path = Path(output_dir) / "cluster_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved cluster analysis summary to: {report_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="PashuRaksha AI Outbreak & Cluster Analytics")
    parser.add_argument("--data", type=str, default="data/clinical/clinical_data.csv", help="Clinical dataset path")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--eps-km", type=float, default=35.0, help="Epsilon neighborhood in km")
    parser.add_argument("--min-samples", type=int, default=4, help="Min samples for cluster core point")
    args = parser.parse_args()

    analyze_disease_clusters(args.data, args.output_dir, args.eps_km, args.min_samples)


if __name__ == "__main__":
    main()
