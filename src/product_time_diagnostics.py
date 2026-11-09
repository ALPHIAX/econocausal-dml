
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("Data/Processed/cleaned_online_retail_dml.csv")

df = pd.read_csv(DATA_PATH)

required = [
    "stock_code", "unit_price", "quantity", "invoicedate"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()

df["stock_code"] = df["stock_code"].astype(str).str.strip()
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["invoicedate"] = pd.to_datetime(
    df["invoicedate"], errors="coerce"
)

df = df.dropna(
    subset=["unit_price", "quantity", "invoicedate"]
)

df = df[
    (df["unit_price"] > 0)
    & (df["quantity"] > 0)
    & (df["stock_code"].str.upper() != "POST")
].copy()

# Reproduce the Stage 2 eligible 100-product cohort
product_stats = df.groupby("stock_code").agg(
    transactions=("stock_code", "size"),
    unique_prices=("unit_price", "nunique")
)

eligible = product_stats[
    (product_stats["transactions"] >= 100)
    & (product_stats["unique_prices"] >= 2)
]

selected_products = (
    eligible.sort_values("transactions", ascending=False)
    .head(100)
    .index
)

df = df[df["stock_code"].isin(selected_products)].copy()

# Create calendar month
df["year_month"] = df["invoicedate"].dt.to_period("M")

print("\n========== COHORT ==========")
print("Observations:", len(df))
print("Products:", df["stock_code"].nunique())
print("Months:", df["year_month"].nunique())

# 1. Product-month summaries
monthly = df.groupby(
    ["stock_code", "year_month"]
).agg(
    transactions=("unit_price", "size"),
    mean_price=("unit_price", "mean"),
    median_price=("unit_price", "median"),
    unique_prices=("unit_price", "nunique"),
    mean_quantity=("quantity", "mean"),
    median_quantity=("quantity", "median")
).reset_index()

print("\n========== PRODUCT-MONTH SUPPORT ==========")
print("Product-month groups:", len(monthly))

for threshold in [5, 10, 20, 50]:
    count = (monthly["transactions"] >= threshold).sum()
    share = 100 * count / len(monthly)
    print(
        f"Groups with >= {threshold} transactions: "
        f"{count} ({share:.2f}%)"
    )

# 2. Product-level temporal variation
product_time = monthly.groupby("stock_code").agg(
    active_months=("year_month", "nunique"),
    total_monthly_transactions=("transactions", "sum"),
    unique_monthly_mean_prices=(
        "mean_price",
        lambda x: x.round(4).nunique()
    ),
    min_monthly_mean_price=("mean_price", "min"),
    max_monthly_mean_price=("mean_price", "max"),
    mean_monthly_quantity=("mean_quantity", "mean"),
    std_monthly_quantity=("mean_quantity", "std")
)

product_time["price_range"] = (
    product_time["max_monthly_mean_price"]
    - product_time["min_monthly_mean_price"]
)

print("\n========== PRODUCT-LEVEL TIME VARIATION ==========")
print(
    product_time[
        [
            "active_months",
            "unique_monthly_mean_prices",
            "price_range",
            "mean_monthly_quantity",
            "std_monthly_quantity"
        ]
    ].describe().round(3)
)

print("\nProducts with at least 2 distinct monthly mean prices:",
      (product_time["unique_monthly_mean_prices"] >= 2).sum())

print("Products with at least 3 distinct monthly mean prices:",
      (product_time["unique_monthly_mean_prices"] >= 3).sum())

# 3. Restrict to product-month groups with at least 10 observations
supported = monthly[monthly["transactions"] >= 10].copy()

print("\n========== SUPPORTED PRODUCT-MONTH GROUPS ==========")
print("Groups with >= 10 transactions:", len(supported))
print("Products represented:", supported["stock_code"].nunique())

# Product-month groups with multiple transaction prices
multiple_price_groups = supported[
    supported["unique_prices"] >= 2
]

print(
    "Supported groups with multiple transaction prices:",
    len(multiple_price_groups)
)

print(
    "Share with multiple prices (%):",
    round(
        100 * len(multiple_price_groups) / len(supported), 2
    ) if len(supported) else 0
)

# 4. Products with strongest and weakest monthly price variation
print("\n========== TOP 10 PRICE-VARYING PRODUCTS ==========")
print(
    product_time.sort_values(
        "price_range", ascending=False
    ).head(10).round(4).to_string()
)

print("\n========== BOTTOM 10 PRICE-VARYING PRODUCTS ==========")
print(
    product_time.sort_values(
        "price_range", ascending=True
    ).head(10).round(4).to_string()
)

# 5. Save detailed diagnostic outputs
output_dir = Path("Data/Processed")
output_dir.mkdir(parents=True, exist_ok=True)

monthly.to_csv(
    output_dir / "product_month_diagnostics.csv",
    index=False
)

product_time.reset_index().to_csv(
    output_dir / "product_time_summary.csv",
    index=False
)

print("\nSaved:")
print("Data/Processed/product_month_diagnostics.csv")
print("Data/Processed/product_time_summary.csv")

print("\nProduct-time diagnostics completed.")