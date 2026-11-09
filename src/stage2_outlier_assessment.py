import pandas as pd
import numpy as np

PATH = "Data/Processed/cleaned_online_retail_dml.csv"

df = pd.read_csv(PATH)
df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")

# Apply the same eligibility criteria used in the previous diagnostics.
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

# Global and product-specific 99th percentile thresholds.
q99 = df["quantity"].quantile(0.99)
p99 = df["unit_price"].quantile(0.99)

df["quantity_p99_product"] = df.groupby("stock_code")["quantity"].transform(
    lambda x: x.quantile(0.99)
)
df["price_p99_product"] = df.groupby("stock_code")["unit_price"].transform(
    lambda x: x.quantile(0.99)
)

df["high_quantity_global"] = df["quantity"] > q99
df["high_price_global"] = df["unit_price"] > p99
df["high_quantity_product"] = df["quantity"] > df["quantity_p99_product"]
df["high_price_product"] = df["unit_price"] > df["price_p99_product"]

print("\n=== 1. COHORT AND THRESHOLDS ===")
print("Rows:", len(df))
print("Products:", df["stock_code"].nunique())
print("Global quantity p99:", q99)
print("Global price p99:", p99)

print("\n=== 2. EXTREME OBSERVATION COUNTS ===")
for col in [
    "high_quantity_global", "high_price_global",
    "high_quantity_product", "high_price_product"
]:
    print(f"{col}: {df[col].sum()} ({df[col].mean():.2%})")

print("\n=== 3. HIGHEST QUANTITY TRANSACTIONS ===")
cols = [
    "invoice_id", "stock_code", "description", "quantity",
    "unit_price", "country", "invoicedate"
]
print(
    df.nlargest(20, "quantity")[cols].to_string(index=False)
)

print("\n=== 4. HIGHEST PRICE TRANSACTIONS ===")
print(
    df.nlargest(20, "unit_price")[cols].to_string(index=False)
)

print("\n=== 5. WHERE EXTREME QUANTITIES OCCUR ===")
extreme_q = df[df["high_quantity_global"]]

print("\nBy product:")
print(
    extreme_q.groupby("stock_code").size()
    .sort_values(ascending=False).head(15).to_string()
)

print("\nBy country:")
print(
    extreme_q.groupby("country").size()
    .sort_values(ascending=False).head(15).to_string()
)

print("\n=== 6. INVOICE-LEVEL CONTEXT ===")
invoice_stats = df.groupby("invoice_id").agg(
    invoice_lines=("stock_code", "size"),
    distinct_products=("stock_code", "nunique"),
    total_quantity=("quantity", "sum"),
    max_line_quantity=("quantity", "max"),
    total_value=("total_amount", "sum")
)

extreme_invoice_ids = df.loc[
    df["high_quantity_global"], "invoice_id"
].drop_duplicates()

print("Invoices containing globally extreme quantities:",
      len(extreme_invoice_ids))

print(
    invoice_stats.loc[extreme_invoice_ids]
    .sort_values("max_line_quantity", ascending=False)
    .head(20).to_string()
)

print("\n=== 7. PRODUCT-LEVEL EXTREME QUANTITY RATES ===")
product_summary = df.groupby("stock_code").agg(
    transactions=("quantity", "size"),
    high_quantity_count=("high_quantity_product", "sum"),
    median_quantity=("quantity", "median"),
    max_quantity=("quantity", "max"),
    median_price=("unit_price", "median"),
    max_price=("unit_price", "max")
)
product_summary["high_quantity_rate"] = (
    product_summary["high_quantity_count"] /
    product_summary["transactions"]
)

print(
    product_summary.sort_values(
        "high_quantity_rate", ascending=False
    ).head(20).to_string()
)

print("\n=== DIAGNOSTIC COMPLETE ===")
