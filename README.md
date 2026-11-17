# Cryptocurrency Clustering — K-Means and PCA

This project uses **unsupervised machine learning** to group 41 cryptocurrencies using seven historical percentage-change features. It includes the original educational Jupyter notebook, a reproducible command-line analysis, regression tests, and GitHub Actions that run the full notebook in a fresh kernel.

**Important:** Clusters are descriptive groupings, not forecasts, price targets, trading signals, or investment recommendations.

## What changed in the cleanup

The old notebook normalized market features but mistakenly fit the first K-means model and PCA using **unscaled** raw data. This made the results sensitive to feature scale and inconsistent with the assignment's described approach.

The notebook now uses a consistent pipeline:

1. Load the historical market CSV, validate identifiers and all numeric features.
2. Standardize each of the seven return features using `StandardScaler`.
3. Fit K-means on standardized features, with an explicit random seed and `n_init=10`.
4. Fit three-component PCA **on the standardized features**, then fit another K-means model on the PCA coordinates.
5. Compare cluster membership, elbow inertia, and explained variance.

The default 4- and 2-cluster settings follow the original notebook demonstration; **they are not objectively proven best** for market forecasting. Cluster IDs are arbitrary and should not be compared numerically across separate models.

## Run on Windows, macOS or Linux

Use Python 3.11 or 3.12 and install dependencies in an isolated virtual environment:

```bash
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

From the repository root:

```bash
python crypto_clustering.py --output-dir reports/latest
```

Files generated in `reports/latest/`:

- `cluster_assignments.csv`: original features, labels from the two models, and PCA coordinates
- `elbow_inertia.csv`: K-means inertia for 1–10 clusters in the standardized and PCA spaces
- `summary.json`: feature names, methodology, chosen `k`, seed, inertia, and PCA explained-variance ratios

Optional parameters include `--input`, `--original-k`, `--pca-k`, and `--seed`.

The CLI resolves the bundled data path relative to its own source file, not the shell's working directory.

## Run the notebook

```bash
cd Starter_Code
jupyter notebook crypto_investments.ipynb
```

Run the cells sequentially. The notebook retains its explanatory exercises and charts but the old saved outputs were cleared because they reflected the inconsistent unscaled computations. The source CSV remains unchanged.

To execute and validate the notebook without overwriting it:

```bash
python scripts/run_notebook.py
```

The notebook executor is meant to be launched from the repository root; it sets the notebook's working directory for access to `Resources/`.

## Tests and verification

```bash
python -m unittest discover -s tests -v
python scripts/run_notebook.py
```

GitHub Actions runs these checks on Python 3.11 and 3.12, plus a CLI smoke test. Unit tests cover the 41-record source dataset, deterministic repeatability, basic PCA properties, CSV validation, invalid cluster counts, and generated report files.

## Limitations and interview discussion

- The CSV is a **historical snapshot**, not a live crypto data feed. No real-time prices are queried.
- K-means assumes Euclidean feature proximity; the observed clusters can change with features, sample composition, scaling, and `k`.
- PCA maximizes explained variance in the sample. It does not guarantee that the resulting components predict anything.
- Unit tests verify deterministic behavior and data constraints, not financial usefulness or a uniquely correct choice of clusters.
- For internship interviews, discuss data validation, the importance of scaling before PCA/K-means, algorithmic reproducibility, and why unsupervised models need careful interpretation.

The original exercise is preserved in [Starter_Code/crypto_investments.ipynb](Starter_Code/crypto_investments.ipynb).
