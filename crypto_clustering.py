"""Reproducible K-means / PCA analysis of a historical cryptocurrency dataset.

Educational clustering: labels describe feature similarity, not future returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "Starter_Code" / "Resources" / "crypto_market_data.csv"


def load_market_data(path: str | Path) -> pd.DataFrame:
    """Require unique coin identifiers and complete finite numeric features."""
    frame = pd.read_csv(path)
    if "coin_id" not in frame.columns:
        raise ValueError("CSV must contain coin_id")
    if len(frame) < 3:
        raise ValueError("At least three coins are required for three-component PCA")
    if frame.columns.duplicated().any():
        raise ValueError("CSV has duplicate column names")
    if frame["coin_id"].isna().any() or not frame["coin_id"].astype(str).str.strip().all():
        raise ValueError("coin_id cannot be empty")
    if frame["coin_id"].duplicated().any():
        raise ValueError("coin_id values must be unique")
    frame = frame.set_index("coin_id")
    if frame.shape[1] < 3:
        raise ValueError("At least three numeric features are required")
    for col in frame:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f"Non-numeric feature: {col}")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Features must not contain missing or infinite values")
    if any(frame[col].nunique() < 2 for col in frame):
        raise ValueError("Every feature must vary across coins")
    return frame


def run_analysis(
    frame: pd.DataFrame,
    original_k: int = 4,
    pca_k: int = 2,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit both models on *standardized* features, using fixed RNG and n_init."""
    n = len(frame)
    if not (1 <= original_k <= n and 1 <= pca_k <= n):
        raise ValueError("Cluster counts must lie between 1 and the number of coins")
    scaled = StandardScaler().fit_transform(frame)
    pca = PCA(n_components=3, svd_solver="full")
    reduced = pca.fit_transform(scaled)

    def cluster(data: np.ndarray, k: int) -> KMeans:
        return KMeans(n_clusters=k, random_state=seed, n_init=10).fit(data)

    first = cluster(scaled, original_k)
    second = cluster(reduced, pca_k)
    table = frame.copy()
    table["original_cluster"] = first.labels_
    table["pca_cluster"] = second.labels_
    for i in range(3):
        table[f"pca_{i + 1}"] = reduced[:, i]
    table.index.name = "coin_id"

    elbow = pd.DataFrame(
        [
            {"space": label, "k": k, "inertia": float(cluster(matrix, k).inertia_)}
            for label, matrix in (("standardized", scaled), ("pca_3d", reduced))
            for k in range(1, min(10, n) + 1)
        ]
    )
    summary = {
        "coins": n,
        "features": list(frame.columns),
        "original_k": original_k,
        "pca_k": pca_k,
        "random_seed": seed,
        "kmeans_n_init": 10,
        "original_inertia": float(first.inertia_),
        "pca_inertia": float(second.inertia_),
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "method": "StandardScaler -> KMeans; StandardScaler -> PCA(3) -> KMeans",
        "caution": "Historical unsupervised labels are not forecasts or investment recommendations.",
    }
    return table, elbow, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/latest"))
    parser.add_argument("--original-k", type=int, default=4)
    parser.add_argument("--pca-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    data = load_market_data(args.input)
    assignments, elbow, summary = run_analysis(
        data, original_k=args.original_k, pca_k=args.pca_k, seed=args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output_dir / "cluster_assignments.csv")
    elbow.to_csv(args.output_dir / "elbow_inertia.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Analyzed {summary['coins']} coins; saved reports in {args.output_dir}")


if __name__ == "__main__":
    main()
