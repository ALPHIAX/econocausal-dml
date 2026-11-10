import pandas as pd
import numpy as np

PATH = "Data/Processed/cleaned_online_retail_dml.csv"

df = pd.read_csv(PATH)
df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")

# Match the previous 100-product, non-POST cohort.
df = df[
    (df["stock_code"] != "POST") &
    (df["unit_price"] > 0) &
    (df["quantity"] > 0)
].dropna(subset=[
    "invoice_id", "stock_code", "quantity",
    "unit_price", "country", "invoicedate"
]).copy()

eligible = (
    df.groupby("stock_code")
    .agg(
        transactions=("quantity", "size"),
        unique_prices=("unit_price", "nunique")
    )
)

eligible = eligible[
    (eligible["transactions"] >= 100) &
    (eligible["unique_prices"] >= 2)
].sort_values("transactions", ascending=False).head(100)

df = df[df["stock_code"].isin(eligible.index)].copy()
df["year_month"] = df["invoicedate"].dt.to_period("M").astype(str)

print("\n=== 1. COHORT AND GLOBAL PRICE DISTRIBUTION ===")
print("Rows:", len(df))
print("Products:", df["stock_code"].nunique())
print(df["unit_price"].describe(
    percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]
).to_string())
print("Unique price values:", df["unit_price"].nunique())

print("\n=== 2. PRODUCT-LEVEL PRICE SUPPORT ===")
product_rows = []

for product, group in df.groupby("stock_code"):
    counts = group["unit_price"].value_counts(normalize=True)
    quantiles = group["unit_price"].quantile([.10, .50, .90])

    product_rows.append({
        "stock_code": product,
        "transactions": len(group),
        "unique_prices": group["unit_price"].nunique(),
        "dominant_price_share": counts.iloc[0],
        "prices_with_5pct_support": (counts >= .05).sum(),
        "price_p10": quantiles.loc[.10],
        "price_median": quantiles.loc[.50],
        "price_p90": quantiles.loc[.90],
        "price_iqr": (
            group["unit_price"].quantile(.75) -
            group["unit_price"].quantile(.25)
        )
    })

product_support = pd.DataFrame(product_rows).set_index("stock_code")

print(product_support.describe().to_string())

print("\nProducts with highest dominant-price concentration:")
print(
    product_support.sort_values(
        "dominant_price_share", ascending=False
    ).head(15).round(3).to_string()
)

print("\nProducts with lowest dominant-price concentration:")
print(
    product_support.sort_values(
        "dominant_price_share", ascending=True
    ).head(15).round(3).to_string()
)

print("\n=== 3. PRODUCT-COUNTRY-MONTH PRICE SUPPORT ===")
group_rows = []

for keys, group in df.groupby(["stock_code", "country", "year_month"]):
    if len(group) < 10:
        continue

    counts = group["unit_price"].value_counts(normalize=True)
    prices = group["unit_price"]

    group_rows.append({
        "stock_code": keys[0],
        "country": keys[1],
        "year_month": keys[2],
        "transactions": len(group),
        "unique_prices": prices.nunique(),
        "dominant_price_share": counts.iloc[0],
        "minority_price_share": 1 - counts.iloc[0],
        "prices_with_5pct_support": (counts >= .05).sum(),
        "price_min": prices.min(),
        "price_p10": prices.quantile(.10),
        "price_median": prices.median(),
        "price_p90": prices.quantile(.90),
        "price_max": prices.max(),
        "price_std": prices.std(),
        "price_cv": (
            prices.std() / prices.mean()
            if prices.mean() != 0 else np.nan
        )
    })

groups = pd.DataFrame(group_rows)

print("Groups with at least 10 transactions:", len(groups))
print("Groups with multiple prices:",
      (groups["unique_prices"] >= 2).sum())
print("Groups with minority share >=5%:",
      (
          (groups["unique_prices"] >= 2) &
          (groups["minority_price_share"] >= .05)
      ).sum())
print("Groups with minority share >=10%:",
      (
          (groups["unique_prices"] >= 2) &
          (groups["minority_price_share"] >= .10)
      ).sum())

print("\nGroup-level price variation summary:")
print(
    groups[[
        "transactions", "unique_prices", "dominant_price_share",
        "minority_price_share", "prices_with_5pct_support",
        "price_p10", "price_median", "price_p90", "price_cv"
    ]].describe().round(3).to_string()
)

print("\n=== 4. EXTREME PRICE FREQUENCY ===")
global_p99 = df["unit_price"].quantile(.99)
extreme_prices = df[df["unit_price"] > global_p99]

print("Global price p99:", global_p99)
print("Observations above p99:", len(extreme_prices))
print("Distinct products above p99:",
      extreme_prices["stock_code"].nunique())

print("\nMost frequent extreme product-price combinations:")
print(
    extreme_prices.groupby(["stock_code", "unit_price"])
    .size()
    .sort_values(ascending=False)
    .head(20).to_string()
)

print("\n=== 5. PRODUCT-LEVEL SUPPORTED-GROUP SHARE ===")
support = groups.assign(
    supported=(
        (groups["unique_prices"] >= 2) &
        (groups["minority_price_share"] >= .05)
    )
).groupby("stock_code").agg(
    comparable_groups=("supported", "size"),
    supported_groups=("supported", "sum")
)

support["supported_group_share"] = (
    support["supported_groups"] / support["comparable_groups"]
)

print(
    support.sort_values("supported_group_share")
    .head(20).round(3).to_string()
)

print("\n=== DIAGNOSTIC COMPLETE ===")
