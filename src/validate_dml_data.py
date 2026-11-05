import pandas as pd


import pandas as pd
from pathlib import Path

# =============================================================
# 1. LOAD DATASET
# =============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "Data" / "Processed" / "cleaned_online_retail_dml.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DML DATASET VALIDATION ==========")

# =============================================================
# 2. STRUCTURAL VALIDATION
# =============================================================

expected_columns = [
    "invoice_id",
    "stock_code",
    "description",
    "quantity",
    "invoicedate",
    "unit_price",
    "customer_id",
    "country",
    "source_sheet",
    "total_amount"
]

print("\n1. STRUCTURAL VALIDATION")

print("Dataset shape:", df.shape)
print("Actual columns:", df.columns.tolist())

missing_columns = set(expected_columns) - set(df.columns)

print("Missing expected columns:", missing_columns)

assert not missing_columns, "Required columns are missing!"

# =============================================================
# 3. MISSING VALUE VALIDATION
# =============================================================

print("\n2. MISSING VALUE VALIDATION")

print(df.isnull().sum())

assert df.isnull().sum().sum() == 0, "Missing values detected!"

# =============================================================
# 4. DUPLICATE VALIDATION
# =============================================================

print("\n3. DUPLICATE VALIDATION")

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)

# =============================================================
# 5. DATA TYPE VALIDATION
# =============================================================

print("\n4. DATA TYPE VALIDATION")

print(df.dtypes)

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"],
    errors="coerce"
)

print("Invalid dates:", df["invoicedate"].isnull().sum())

# =============================================================
# 6. BUSINESS RULE VALIDATION
# =============================================================

print("\n5. BUSINESS RULE VALIDATION")

print("Non-positive quantities:", (df["quantity"] <= 0).sum())

print("Non-positive prices:", (df["unit_price"] <= 0).sum())

print(
    "Cancelled invoices:",
    df["invoice_id"].astype(str).str.upper().str.startswith("C").sum()
)

print("Missing customer IDs:", df["customer_id"].isnull().sum())

# =============================================================
# 7. FEATURE VALIDATION
# =============================================================

print("\n6. FEATURE VALIDATION")

calculated_amount = df["quantity"] * df["unit_price"]

amount_mismatches = (
    (df["total_amount"] - calculated_amount).abs() > 0.000001
).sum()

print("Total amount mismatches:", amount_mismatches)

# =============================================================
# 8. STATISTICAL VALIDATION
# =============================================================

print("\n7. NUMERICAL SUMMARY")

print(
    df[["quantity", "unit_price", "total_amount"]].describe()
)

print("\n8. CATEGORICAL SUMMARY")

print("Unique products:", df["stock_code"].nunique())
print("Unique customers:", df["customer_id"].nunique())
print("Unique countries:", df["country"].nunique())
print("Unique invoices:", df["invoice_id"].nunique())

# =============================================================
# 9. FINAL VALIDATION STATUS
# =============================================================

print("\n========== FINAL VALIDATION ==========")

checks = {
    "Required columns present": not missing_columns,
    "No missing values": df.isnull().sum().sum() == 0,
    "No duplicate records": duplicate_count == 0,
    "No invalid dates": df["invoicedate"].notnull().all(),
    "Positive quantities": (df["quantity"] > 0).all(),
    "Positive prices": (df["unit_price"] > 0).all(),
    "No cancelled invoices": ~df["invoice_id"].astype(str).str.upper().str.startswith("C").any(),
    "No missing customer IDs": df["customer_id"].notnull().all(),
    "Correct total amount": amount_mismatches == 0
}

for check, status in checks.items():
    print(f"{'PASS' if status else 'FAIL'}: {check}")

print("\nPassed:", sum(checks.values()), "/", len(checks))

if all(checks.values()):
    print("\nDATASET VALIDATION PASSED")
else:
    print("\nDATASET VALIDATION NEEDS ATTENTION")
