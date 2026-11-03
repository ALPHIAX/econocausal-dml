import pandas as pd
import numpy as np

INPUT_PATH = "Data/Processed/dml_modeling_data.csv"
OUTPUT_PATH = "Data/Processed/product_price_support.csv"

df = pd.read_csv(INPUT_PATH)

# Validate required columns
required = ["stock_code", "unit_price", "quantity"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Remove invalid treatment observations
df = df.dropna(subset=["stock_code", "unit_price"])
df = df[np.isfinite(df["unit_price"])]

# Product-level price variation
support = (
    df.groupby("stock_code")["unit_price"]
    .agg(
        transaction_count="count",
        unique_prices="nunique",
        min_price="min",
        max_price="max",
        mean_price="mean",
        std_price="std"
    )
    .reset_index()
)

support["price_range"] = (
    support["max_price"] - support["min_price"]
)

support["has_price_variation"] = (
    support["unique_prices"] > 1
)

# Overall treatment distribution
print("=== PRICE SUPPORT DIAGNOSTICS ===")
print("Observations:", len(df))
print("Unique products:", support.shape[0])

print("\nPrice distribution:")
print(df["unit_price"].describe(
    percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]
))

print("\nProduct-level price variation:")
print("Products with multiple prices:",
      support["has_price_variation"].sum())

print("Products with one observed price:",
      (~support["has_price_variation"]).sum())

print("\nTransactions from products with multiple prices:")
variable_products = set(
    support.loc[
        support["has_price_variation"], "stock_code"
    ]
)

variable_transactions = df["stock_code"].isin(
    variable_products
).sum()

print(variable_transactions)
print("Percentage:",
      round(variable_transactions / len(df) * 100, 2), "%")

# Save diagnostic evidence
support.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)