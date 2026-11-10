"""Prepare a descriptive CATE-ranking dataset from held-out DML estimates.

This is NOT a Qini/uplift evaluation. The treatment is continuous (unit_price),
and the available data are observational transaction records. This script only
prepares model-estimated CATEs for a transparent ranking diagnostic.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "Data" / "Processed" / "dml_ite_results.csv"
OUTPUT_PATH = ROOT / "Data" / "Processed" / "cate_ranking_data.csv"

REQUIRED_COLUMNS = {
    "invoice_id",
    "stock_code",
    "unit_price",
    "quantity",
    "estimated_cate",
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Required DML results file not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"DML results are missing required columns: {sorted(missing)}")

    df = df[list(REQUIRED_COLUMNS)].copy()
    for column in ["unit_price", "quantity", "estimated_cate"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["unit_price", "quantity", "estimated_cate"])
    df = df[(df["unit_price"] > 0) & (df["quantity"] > 0)].copy()

    if df.empty:
        raise ValueError("No valid DML estimates remain after validation.")

    df.to_csv(OUTPUT_PATH, index=False)

    print("=== CATE Ranking Data Preparation ===")
    print("Input rows:", before)
    print("Valid rows:", len(df))
    print("Rows removed:", before - len(df))
    print("Unique products:", df["stock_code"].nunique())
    print("Mean estimated CATE:", round(df["estimated_cate"].mean(), 6))
    print("Saved:", OUTPUT_PATH)
    print("Interpretation: descriptive model-output diagnostic only; not Qini/uplift.")


if __name__ == "__main__":
    main()

