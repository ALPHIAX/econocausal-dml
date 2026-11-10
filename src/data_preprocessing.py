
import pandas as pd
from pathlib import Path


# =============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# =============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "Data" / "Raw" / "online_retail_II.xlsx"
OUTPUT_DIR = ROOT / "Data" / "Processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_OUTPUT = OUTPUT_DIR / "online_retail_cleaned.csv"
DML_OUTPUT = OUTPUT_DIR / "cleaned_online_retail_dml.csv"


# =============================================================
# 2. LOAD ONLINE RETAIL II DATASET
# =============================================================

print("\n--- Starting EconoCausal Data Preprocessing ---")

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f"Online Retail II dataset not found: {RAW_FILE}"
    )

print("\nLoading Online Retail II Excel workbook...")

sheets = pd.read_excel(RAW_FILE, sheet_name=None)

frames = []

for sheet_name, sheet_df in sheets.items():
    print(f"Reading sheet: {sheet_name} | Shape: {sheet_df.shape}")

    sheet_df["source_sheet"] = sheet_name
    frames.append(sheet_df)

if not frames:
    raise ValueError("No data found in the Excel workbook.")

retail = pd.concat(frames, ignore_index=True)

print(f"\nCombined dataset shape: {retail.shape}")


# =============================================================
# 3. STANDARDIZE COLUMN NAMES
# =============================================================

retail.columns = (
    retail.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

required_columns = [
    "invoice",
    "stockcode",
    "description",
    "quantity",
    "invoicedate",
    "price",
    "customer_id",
    "country"
]

missing_columns = [
    col for col in required_columns
    if col not in retail.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =============================================================
# 4. DATA TYPE CONVERSION
# =============================================================

print("\nConverting data types...")

retail["invoicedate"] = pd.to_datetime(
    retail["invoicedate"],
    errors="coerce"
)

retail["quantity"] = pd.to_numeric(
    retail["quantity"],
    errors="coerce"
)

retail["price"] = pd.to_numeric(
    retail["price"],
    errors="coerce"
)

retail["customer_id"] = pd.to_numeric(
    retail["customer_id"],
    errors="coerce"
)

retail["invoice"] = retail["invoice"].astype("string").str.strip()

retail["stockcode"] = retail["stockcode"].astype("string").str.strip()

retail["description"] = (
    retail["description"]
    .astype("string")
    .str.strip()
)

retail["country"] = (
    retail["country"]
    .astype("string")
    .str.strip()
)


# =============================================================
# 5. GENERAL CLEANING
# =============================================================

print("\nCleaning general dataset...")

retail = retail.drop_duplicates().copy()

retail["description"] = retail["description"].fillna("UNKNOWN")
retail["country"] = retail["country"].fillna("UNKNOWN")

# Save the generally cleaned dataset.
retail.to_csv(CLEANED_OUTPUT, index=False)

print(f"General cleaned dataset saved: {CLEANED_OUTPUT}")
print(f"General cleaned shape: {retail.shape}")


# =============================================================
# 6. DML-SPECIFIC DATA CLEANING
# =============================================================

print("\nPreparing DML modeling dataset...")

df_dml = retail.copy()

# Rule 1: Identify cancelled invoices and non-positive quantities.
is_cancellation = (
    df_dml["invoice"].str.upper().str.startswith("C", na=False)
    | (df_dml["quantity"] <= 0)
)

# IMPORTANT: Apply the cancellation filter.
df_dml = df_dml[~is_cancellation].copy()

# Rule 2: Keep only positive unit prices.
df_dml = df_dml[df_dml["price"] > 0].copy()

# Rule 3: Remove rows with missing essential values.
df_dml = df_dml.dropna(
    subset=[
        "invoice",
        "stockcode",
        "quantity",
        "invoicedate",
        "price",
        "customer_id"
    ]
).copy()

# Rule 4: Convert customer IDs to integers.
df_dml["customer_id"] = df_dml["customer_id"].astype("int64")

# Rule 5: Handle missing descriptions and countries.
df_dml["description"] = df_dml["description"].fillna("UNKNOWN")
df_dml["country"] = df_dml["country"].fillna("UNKNOWN")

# Rule 6: Remove duplicates after cleaning.
df_dml = df_dml.drop_duplicates().copy()


# =============================================================
# 7. FEATURE ENGINEERING AND COLUMN RENAMING
# =============================================================

print("\nEngineering DML variables...")

# Calculate total transaction amount.
df_dml["total_amount"] = (
    df_dml["quantity"] * df_dml["price"]
)

# Standardize names for downstream DML scripts.
df_dml = df_dml.rename(
    columns={
        "price": "unit_price",
        "invoice": "invoice_id",
        "stockcode": "stock_code"
    }
)


# =============================================================
# 8. SAVE DML MODELING DATASET
# =============================================================

df_dml.to_csv(DML_OUTPUT, index=False)

print(f"\nDML dataset saved: {DML_OUTPUT}")
print(f"DML dataset shape: {df_dml.shape}")


# =============================================================
# 9. FINAL DATA QUALITY CHECKS
# =============================================================

print("\n--- DML Dataset Validation Summary ---")

print(f"Total rows: {len(df_dml)}")
print(f"Total columns: {len(df_dml.columns)}")

print("\nMissing values:")
print(df_dml.isnull().sum())

print("\nBasic checks:")
print(
    "Cancelled invoices remaining:",
    df_dml["invoice_id"].astype(str).str.upper().str.startswith("C").sum()
)

print(
    "Non-positive quantities:",
    (df_dml["quantity"] <= 0).sum()
)

print(
    "Non-positive unit prices:",
    (df_dml["unit_price"] <= 0).sum()
)

print(
    "Missing customer IDs:",
    df_dml["customer_id"].isnull().sum()
)

print("\n--- Preprocessing Completed Successfully ---")
