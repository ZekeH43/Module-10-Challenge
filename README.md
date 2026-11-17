# Crypto Clustering with K-Means and PCA

This is a preserved **FinTech course notebook**, updated with a project overview and basic source-integrity checks. The original assignment code remains in [Starter_Code/crypto_investments.ipynb](Starter_Code/crypto_investments.ipynb).

## Purpose

The exercise uses Python to scale cryptocurrency market features, compare elbow curves, cluster with K-means, and compare clusters before and after principal component analysis. It is intended as a learning artifact and **has not been revalidated as a production or live-data application**.

## Repository layout

- Main notebook: `Starter_Code/crypto_investments.ipynb`
- Source data: Starter_Code/Resources/crypto_market_data.csv
- `scripts/validate_assets.py`: built-in Python checks that notebooks are readable JSON and included CSV files have headers
- `.github/workflows/source-integrity.yml`: runs the source checks on pull requests

## Running locally

1. Use an isolated Python environment suitable for the historical notebook. Start with Python 3.10/3.11, adjusting packages as necessary.
2. From the repository root, optionally create a virtual environment:
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux: source .venv/bin/activate
   python -m pip install notebook pandas scikit-learn hvplot
   ```
3. Launch Jupyter with the notebook directory as the working directory:
   ```bash
   cd Starter_Code
   jupyter notebook crypto_investments.ipynb
   ```
4. Review the notebook's imports, file paths, package versions, and any external-data requirements before running cells in order.

The installation command is an **unverified starting point**, not a pinned or tested dependency set. Historic data formats and package APIs may need additional work.

## Lightweight repository validation

Run from the repository root:

```bash
python scripts/validate_assets.py
```

This checks file integrity only: it **does not execute notebooks, train models, test analytics, or validate numerical outcomes**.

## Limitations and portfolio context

Unsupervised clusters summarize the historical input features; they do not predict price performance or establish investment suitability.

For an internship discussion, explain the data preparation, methods selected, evaluation limits, and what you would improve for reliable reproducibility. Original notebooks and assignment resources remain preserved.
