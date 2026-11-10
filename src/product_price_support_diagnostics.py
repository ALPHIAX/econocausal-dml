"""Reproducible support diagnostics for product- and time-level price variation.

Reads the engineered DML dataset and writes three audit artifacts:
1. A one-row summary of product and product-month price support.
2. Product-level transaction and distinct-price counts.
3. Product-month-level transaction and distinct-price counts.

This is a descriptive support diagnostic, not a causal-effect estimator.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "Data" / "Processed" / "dml_features.csv"
OUTPUT_DIR = ROOT / "Data" / "Processed"

# Descriptive thresholds only; these are not claims of statistical sufficiency.
PRODUCT_MIN_TRANSACTIONS = 100
PRODUCT_MONTH_MIN_TRANSACTIONS = (10, 20)

REQUIRED_COLUMNS = {
    "stock_code",
    "unit_price",
    "year",
    "month_sin",
    "month_cos",
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"DML feature dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("DML feature dataset is empty.")

    if df[["stock_code", "unit_price", "year", "month_sin", "month_cos"]].isna().any().any():
        raise ValueError("Null values found in product, price, or month grouping columns.")
    if (df["unit_price"] <= 0).any():
        raise ValueError("Non-positive unit prices found; resolve data quality before support analysis.")

    # Month is represented by a deterministic sine/cosine pair. Grouping on
    # both values preserves calendar-month identity within each year.
    product = (
        df.groupby("stock_code", observed=True)
        .agg(
            transactions=("unit_price", "size"),
            unique_prices=("unit_price", "nunique"),
        )
        .reset_index()
    )
    product["has_multiple_prices"] = product["unique_prices"] >= 2
    product["has_multiple_prices_and_min_transactions"] = (
        product["has_multiple_prices"]
        & (product["transactions"] >= PRODUCT_MIN_TRANSACTIONS)
    )

    product_month = (
        df.groupby(
            ["stock_code", "year", "month_sin", "month_cos"],
            observed=True,
            dropna=False,
        )
        .agg(
            transactions=("unit_price", "size"),
            unique_prices=("unit_price", "nunique"),
        )
        .reset_index()
    )
    product_month["has_multiple_prices"] = product_month["unique_prices"] >= 2
    for threshold in PRODUCT_MONTH_MIN_TRANSACTIONS:
        product_month[f"multiple_prices_and_{threshold}_transactions"] = (
            product_month["has_multiple_prices"]
            & (product_month["transactions"] >= threshold)
        )

    n_rows = len(df)
    n_products = len(product)
    n_product_months = len(product_month)

    summary = pd.DataFrame(
        [
            {
                "transactions": n_rows,
                "products": n_products,
                "products_with_multiple_prices": int(product["has_multiple_prices"].sum()),
                "products_with_multiple_prices_pct": round(
                    100 * product["has_multiple_prices"].mean(), 2
                ),
                "products_with_multiple_prices_and_100_transactions": int(
                    product["has_multiple_prices_and_min_transactions"].sum()
                ),
                "product_month_groups": n_product_months,
                "product_months_with_multiple_prices": int(
                    product_month["has_multiple_prices"].sum()
                ),
                "product_months_with_multiple_prices_pct": round(
                    100 * product_month["has_multiple_prices"].mean(), 2
                ),
                "product_months_with_multiple_prices_and_10_transactions": int(
                    product_month["multiple_prices_and_10_transactions"].sum()
                ),
                "product_months_with_multiple_prices_and_20_transactions": int(
                    product_month["multiple_prices_and_20_transactions"].sum()
                ),
                "product_month_transaction_median": float(
                    product_month["transactions"].median()
                ),
                "product_month_unique_price_median": float(
                    product_month["unique_prices"].median()
                ),
                "product_min_transactions_threshold": PRODUCT_MIN_TRANSACTIONS,
                "product_month_min_transactions_thresholds": "10,20",
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "product_price_support_summary.csv"
    product_path = OUTPUT_DIR / "product_price_support_by_product.csv"
    product_month_path = OUTPUT_DIR / "product_price_support_by_product_month.csv"

    summary.to_csv(summary_path, index=False)
    product.to_csv(product_path, index=False)
    product_month.to_csv(product_month_path, index=False)

    print("=== Product Price Support Diagnostics ===")
    print(f"Input rows: {n_rows:,}")
    print(f"Products: {n_products:,}")
    print(
        "Products with multiple prices: "
        f"{int(product['has_multiple_prices'].sum()):,} "
        f"({100 * product['has_multiple_prices'].mean():.1f}%)"
    )
    print(f"Product-month groups: {n_product_months:,}")
    print(
        "Product-month groups with multiple prices: "
        f"{int(product_month['has_multiple_prices'].sum()):,} "
        f"({100 * product_month['has_multiple_prices'].mean():.1f}%)"
    )
    for threshold in PRODUCT_MONTH_MIN_TRANSACTIONS:
        count = int(product_month[f"multiple_prices_and_{threshold}_transactions"].sum())
        print(f"Multiple prices and {threshold}+ transactions: {count:,}")

    print("\nSaved diagnostics:")
    print(summary_path)
    print(product_path)
    print(product_month_path)
    print("\nNote: transaction thresholds are descriptive filters, not causal-identification guarantees.")


if __name__ == "__main__":
    main()
