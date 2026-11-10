
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("Data/Processed/cleaned_online_retail_dml.csv")

df = pd.read_csv(DATA_PATH)

# Validate and prepare the required columns
required = [
    "customer_id", "stock_code", "unit_price",
    "quantity", "invoice_id"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()

df["stock_code"] = df["stock_code"].astype(str).str.strip()
df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

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

print("\n========== COHORT ==========")
print("Observations:", len(df))
print("Products:", df["stock_code"].nunique())
print("Customers:", df["customer_id"].nunique())

# 1. Customer repeat-transaction behaviour
customer_stats = df.groupby("customer_id").agg(
    transactions=("invoice_id", "size"),
    unique_products=("stock_code", "nunique"),
    average_quantity=("quantity", "mean"),
    average_price=("unit_price", "mean")
)

print("\n========== CUSTOMER REPEAT PATTERNS ==========")
print("Customers:", len(customer_stats))
print("Single-transaction customers:",
      (customer_stats["transactions"] == 1).sum())
print("Repeat-transaction customers:",
      (customer_stats["transactions"] > 1).sum())

print("\nTransactions per customer:")
print(customer_stats["transactions"].describe(
    percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
).round(3))

# 2. Price variation within the same customer-product pair
pair_stats = df.groupby(
    ["customer_id", "stock_code"]
).agg(
    transactions=("unit_price", "size"),
    unique_prices=("unit_price", "nunique"),
    price_std=("unit_price", "std"),
    min_price=("unit_price", "min"),
    max_price=("unit_price", "max")
).reset_index()

repeat_pairs = pair_stats[pair_stats["transactions"] > 1]

print("\n========== CUSTOMER-PRODUCT PRICE VARIATION ==========")
print("Customer-product pairs:", len(pair_stats))
print("Repeated pairs:", len(repeat_pairs))
print(
    "Repeated pairs with multiple prices:",
    (repeat_pairs["unique_prices"] > 1).sum()
)
print(
    "Share of repeated pairs with multiple prices (%):",
    round(
        100 * (repeat_pairs["unique_prices"] > 1).mean(), 2
    ) if len(repeat_pairs) else 0
)

# 3. Customer-level variation in price and quantity
print("\n========== CUSTOMER-LEVEL SUMMARY ==========")
print(
    customer_stats[
        ["average_price", "average_quantity", "unique_products"]
    ].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]
    ).round(3)
)

# 4. How many customers have enough repeated observations?
print("\n========== CUSTOMER SUPPORT ==========")
for threshold in [2, 5, 10, 20, 50]:
    count = (customer_stats["transactions"] >= threshold).sum()
    share = 100 * count / len(customer_stats)
    print(
        f"Customers with >= {threshold} transactions: "
        f"{count} ({share:.2f}%)"
    )

print("\nCustomer diagnostics completed.")
