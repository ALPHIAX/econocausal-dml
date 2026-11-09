from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "Data" / "Processed" / "cleaned_online_retail_dml.csv"

df = pd.read_csv(INPUT_PATH)

required = [
    "stock_code", "quantity", "unit_price", "invoicedate"
]

df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
df["stock_code"] = df["stock_code"].astype(str).str.strip()

df = df.dropna(subset=required)
df = df[
    (df["unit_price"] > 0) &
    (df["quantity"] > 0) &
    (df["stock_code"].str.upper() != "POST")
].copy()

# Reproduce the agreed top-100 product cohort
support = df.groupby("stock_code").agg(
    transactions=("unit_price", "size"),
    unique_prices=("unit_price", "nunique")
)

eligible = support[
    (support["transactions"] >= 100) &
    (support["unique_prices"] >= 2)
]

selected = eligible.sort_values(
    "transactions", ascending=False
).head(100).index

df = df[df["stock_code"].isin(selected)].copy()

# Product-month groups
df["year_month"] = df["invoicedate"].dt.to_period("M").astype(str)

groups = df.groupby(["stock_code", "year_month"]).agg(
    n=("unit_price", "size"),
    unique_prices=("unit_price", "nunique"),
    price_min=("unit_price", "min"),
    price_max=("unit_price", "max"),
    price_std=("unit_price", "std"),
    price_mean=("unit_price", "mean")
).reset_index()

groups["price_cv"] = (
    groups["price_std"] / groups["price_mean"]
)

supported = groups[groups["n"] >= 10].copy()

supported["multiple_prices"] = supported["unique_prices"] >= 2
supported["price_cv_positive"] = supported["price_std"] > 0

# Product-level support
product_support = supported.groupby("stock_code").agg(
    supported_months=("year_month", "count"),
    months_with_multiple_prices=("multiple_prices", "sum"),
    mean_price_cv=("price_cv", "mean"),
    median_price_cv=("price_cv", "median")
)

product_support["multiple_price_month_share"] = (
    product_support["months_with_multiple_prices"] /
    product_support["supported_months"]
)

# Price distribution within product-month groups
print("\n========== STAGE 6: OVERLAP DIAGNOSTICS ==========")
print("Cohort observations:", len(df))
print("Products:", df["stock_code"].nunique())
print("Product-month groups:", len(groups))
print("Groups with >=10 observations:", len(supported))

print("\n========== WITHIN-GROUP PRICE SUPPORT ==========")
print("Groups with multiple prices:",
      int(supported["multiple_prices"].sum()))
print("Share with multiple prices:",
      round(supported["multiple_prices"].mean(), 4))
print("Groups with zero price variation:",
      int((~supported["multiple_prices"]).sum()))

print("\nUnique prices per supported group:")
print(supported["unique_prices"].describe())

print("\nPrice CV across supported groups:")
print(supported["price_cv"].describe())

print("\n========== PRODUCT-LEVEL SUPPORT ==========")
print(product_support.describe())

print("\nProducts with weakest multiple-price support:")
print(
    product_support.sort_values(
        "multiple_price_month_share"
    ).head(15).to_string()
)

print("\nProducts with strongest multiple-price support:")
print(
    product_support.sort_values(
        "multiple_price_month_share",
        ascending=False
    ).head(10).to_string()
)

# Overall within-group price support
print("\n========== PRICE RANGE CHECK ==========")
print("Overall price range:", df["unit_price"].min(),
      "to", df["unit_price"].max())

print("Supported-group price range:")
print(supported["price_min"].min(),
      "to", supported["price_max"].max())

print("Groups with zero/undefined price CV:",
      int(supported["price_cv"].isna().sum() +
          (supported["price_cv"] == 0).sum()))

# Descriptive validation only; not a causal overlap pass/fail test
checks = {
    "100 products represented": df["stock_code"].nunique() == 100,
    "Supported product-month groups exist": len(supported) > 0,
    "Some supported groups have multiple prices":
        supported["multiple_prices"].any(),
    "No POST product": not df["stock_code"].str.upper().eq("POST").any()
}

print("\n========== BASIC VALIDATION ==========")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

if not all(checks.values()):
    raise SystemExit("Basic overlap diagnostic validation failed.")

print("\nSTAGE 6 DIAGNOSTICS COMPLETED")
