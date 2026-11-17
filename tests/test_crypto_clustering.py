"""Deterministic analysis and data-validation regression tests."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_clustering import DATA_PATH, load_market_data, run_analysis


class CryptoClusteringTests(unittest.TestCase):
    def write_csv(self, rows, header=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "market.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header or ["coin_id", "feature_a", "feature_b", "feature_c"])
            writer.writerows(rows)
        return path

    def test_original_data_is_valid(self):
        frame = load_market_data(DATA_PATH)
        self.assertEqual(frame.shape, (41, 7))
        self.assertIn("bitcoin", frame.index)
        self.assertFalse(frame.isna().any().any())

    def test_reproducible_clusters_and_pca(self):
        frame = load_market_data(DATA_PATH)
        table_a, elbow_a, summary_a = run_analysis(frame)
        table_b, elbow_b, summary_b = run_analysis(frame)
        pd.testing.assert_frame_equal(table_a, table_b)
        pd.testing.assert_frame_equal(elbow_a, elbow_b)
        self.assertEqual(summary_a, summary_b)
        self.assertEqual(table_a["original_cluster"].nunique(), 4)
        self.assertEqual(table_a["pca_cluster"].nunique(), 2)
        self.assertEqual(len(elbow_a), 20)
        self.assertTrue(np.isfinite(elbow_a["inertia"].to_numpy()).all())
        self.assertTrue(0 < sum(summary_a["pca_explained_variance_ratio"]) <= 1.000001)

    def test_standardized_space_differs_from_raw_features(self):
        frame = load_market_data(DATA_PATH)
        _, _, summary = run_analysis(frame)
        self.assertIn("StandardScaler -> KMeans", summary["method"])
        self.assertIn("StandardScaler -> PCA", summary["method"])

    def test_rejects_missing_or_duplicate_coin_ids(self):
        for rows in [
            [["same", 1, 2, 3], ["same", 4, 5, 6], ["third", 7, 8, 9]],
            [["", 1, 2, 3], ["two", 4, 5, 6], ["three", 7, 8, 9]],
        ]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_market_data(self.write_csv(rows))

    def test_rejects_bad_features(self):
        cases = [
            [["a", 1, 2, 3], ["b", "oops", 5, 6], ["c", 7, 8, 9]],
            [["a", 1, 2, 3], ["b", "", 5, 6], ["c", 7, 8, 9]],
            [["a", 1, 2, 3], ["b", 1, 5, 6], ["c", 1, 8, 9]],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_market_data(self.write_csv(rows))

    def test_rejects_invalid_k(self):
        frame = load_market_data(DATA_PATH)
        for raw, pca in [(0, 2), (2, 0), (42, 2), (4, 42)]:
            with self.subTest(k=(raw, pca)), self.assertRaises(ValueError):
                run_analysis(frame, original_k=raw, pca_k=pca)

    def test_cli_writes_parseable_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "reports"
            subprocess.run(
                [sys.executable, "crypto_clustering.py", "--output-dir", str(output)],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
            assignments = pd.read_csv(output / "cluster_assignments.csv")
            elbow = pd.read_csv(output / "elbow_inertia.csv")
            meta = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(assignments), 41)
            self.assertEqual(len(elbow), 20)
            self.assertEqual(meta["coins"], 41)


if __name__ == "__main__":
    unittest.main()
