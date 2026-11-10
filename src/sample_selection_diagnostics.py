from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "Data" / "Processed" / "cleaned_online_retail_dml.csv"

df = pd.read_csv(INPUT_PATH)

required = [
    "invoice_id", "stock_code", "quantity",
    "invoicedate", "unit_price", "country"
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
df["stock_code"] = df["stock_code"].astype(str).str.strip()

df = df.dropna(subset=required)
df = df[(df["unit_price"] > 0) & (df["quantity"] > 0)].copy()
df = df[df["stock_code"].str.upper() != "POST"].copy()

support = df.groupby("stock_code").agg(
    transactions=("quantity", "size"),
    unique_prices=("unit_price", "nunique")
)

eligible = support[
    (support["transactions"] >= 100) &
    (support["unique_prices"] >= 2)
]

selected = eligible.sort_values(
    "transactions", ascending=False
).head(100).index

cohort = df[df["stock_code"].isin(selected)].copy()
cohort.reset_index(drop=True, inplace=True)

if cohort["stock_code"].nunique() != 100:
    raise ValueError("Expected exactly 100 selected products.")

if cohort["stock_code"].str.upper().eq("POST").any():
    raise ValueError("POST found in selected cohort.")

indices = np.arange(len(cohort))
train_idx, test_idx = train_test_split(
    indices, test_size=0.20, random_state=42
)

product_stats = cohort.groupby("stock_code").agg(
    transactions=("quantity", "size"),
    unique_prices=("unit_price", "nunique")
)

print("\n========== STAGE 5: SAMPLE SELECTION ==========")
print("Raw rows:", len(pd.read_csv(INPUT_PATH)))
print("Eligible products:", len(eligible))
print("Selected products:", cohort["stock_code"].nunique())
print("Selected observations:", len(cohort))
print("POST present:", cohort["stock_code"].str.upper().eq("POST").any())

print("\n========== COHORT PROFILE ==========")
print("Date range:", cohort["invoicedate"].min(), "to", cohort["invoicedate"].max())
print("Countries:", cohort["country"].nunique())
print("Products per country:\n", cohort["country"].value_counts().head(10))
print("Transactions per product:")
print(product_stats["transactions"].describe())
print("Unique prices per product:")
print(product_stats["unique_prices"].describe())

print("\n========== TREATMENT / OUTCOME ==========")
print("Price summary:\n", cohort["unit_price"].describe())
print("Quantity summary:\n", cohort["quantity"].describe())
print("Unique treatment values:", cohort["unit_price"].nunique())
print("Products with price variation:",
      (product_stats["unique_prices"] >= 2).sum())

print("\n========== TEMPORAL COVERAGE ==========")
print("Observations by year:\n", cohort["invoicedate"].dt.year.value_counts().sort_index())
print("Observations by month:\n", cohort["invoicedate"].dt.to_period("M").value_counts().sort_index())

print("\n========== EXTERNAL SPLIT ==========")
print("Training rows:", len(train_idx))
print("Test rows:", len(test_idx))
print("Train fraction:", round(len(train_idx) / len(cohort), 4))
print("Test fraction:", round(len(test_idx) / len(cohort), 4))

print("\n========== VALIDATION ==========")
checks = {
    "Exactly 100 products": cohort["stock_code"].nunique() == 100,
    "No POST product": not cohort["stock_code"].str.upper().eq("POST").any(),
    "All prices positive": (cohort["unit_price"] > 0).all(),
    "All quantities positive": (cohort["quantity"] > 0).all(),
    "No missing required values": not cohort[required].isna().any().any(),
    "All selected products have >=100 rows": (product_stats["transactions"] >= 100).all(),
    "All selected products have >=2 prices": (product_stats["unique_prices"] >= 2).all(),
    "Split covers all observations": len(train_idx) + len(test_idx) == len(cohort),
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

if not all(checks.values()):
    raise SystemExit("Sample-selection validation failed.")

print("\nSTAGE 5 VALIDATION PASSED")