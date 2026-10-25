import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

print("--- Starting EconoCausal Data Preprocessing ---")

# -------------------------------------------------------------
# 1. Load & Clean Starbucks Datasets
# -------------------------------------------------------------
portfolio = pd.read_json('Data/Raw/portfolio.json', orient='records', lines=True)
profile = pd.read_json('Data/Raw/profile.json', orient='records', lines=True)
transcript = pd.read_json('Data/Raw/transcript.json', orient='records', lines=True)

print("Starbucks datasets loaded successfully!")

# Helper function to convert unhashable types (lists, dicts) to strings for deduplication
def stringify_complex_cols(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    return df

# Clean Profile
profile.drop_duplicates(inplace=True)
profile['age'] = profile['age'].replace(118, np.nan)
profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d', errors='coerce')

# Clean Portfolio
portfolio = stringify_complex_cols(portfolio)
portfolio.drop_duplicates(inplace=True)

# Clean Transcript
transcript = stringify_complex_cols(transcript)
transcript.drop_duplicates(inplace=True)

# Create Processed Directory
os.makedirs('Data/Processed', exist_ok=True)

# Save Cleaned Starbucks Datasets to CSV
profile.to_csv('Data/Processed/cleaned_profile.csv', index=False)
portfolio.to_csv('Data/Processed/cleaned_portfolio.csv', index=False)
transcript.to_csv('Data/Processed/cleaned_transcript.csv', index=False)
print("Starbucks processed files saved to Data/Processed/")


# -------------------------------------------------------------
# 2. Online Retail II Preprocessing (General & DML Modeling-Ready)
# -------------------------------------------------------------
print("\n--- Starting Online Retail II Preprocessing ---")

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "Data" / "Raw" / "online_retail_II.xlsx"
OUTPUT_DIR = ROOT / "Data" / "Processed"

# Validate the raw dataset before loading
if not RAW_FILE.exists():
    raise FileNotFoundError(f"Online Retail II dataset not found: {RAW_FILE}")

# Load all worksheets in the Excel workbook
sheets = pd.read_excel(RAW_FILE, sheet_name=None)
print(f"Number of worksheets loaded: {len(sheets)}")

frames = []
for sheet_name, frame in sheets.items():
    print(f"Loaded {sheet_name}: {frame.shape}")
    frame["source_sheet"] = sheet_name
    frames.append(frame)

retail = pd.concat(frames, ignore_index=True)

# Standardize column names
retail.columns = (
    retail.columns.astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

# Convert key columns to appropriate types
retail["invoicedate"] = pd.to_datetime(retail["invoicedate"], errors="coerce")
retail["quantity"] = pd.to_numeric(retail["quantity"], errors="coerce")
retail["price"] = pd.to_numeric(retail["price"], errors="coerce")
retail["customer_id"] = pd.to_numeric(retail["customer_id"], errors="coerce")

# Save general cleaned retail file
retail.drop_duplicates().to_csv(OUTPUT_DIR / "online_retail_cleaned.csv", index=False)


# -------------------------------------------------------------
# 3. DML Modeling-Ready Filtering & Export
# -------------------------------------------------------------
print("\n--- Generating DML Modeling-Ready Retail Dataset ---")

df_dml = retail.copy()

# Rule 1: Remove cancellations (Invoice starts with 'C') and non-positive quantities
is_cancellation = df_dml['invoice'].astype(str).str.startswith('c') | (df_dml['quantity'] <= 0)
df_dml = df_dml[~is_cancellation].copy()

# Rule 2: Keep price > 0
df_dml = df_dml[df_dml['price'] > 0].copy()

# Rule 3: Drop rows with missing Customer ID
df_dml = df_dml.dropna(subset=['customer_id']).copy()
df_dml['customer_id'] = df_dml['customer_id'].astype(int)

# Rule 4: Handle missing descriptions
df_dml['description'] = df_dml['description'].fillna('UNKNOWN').str.strip()

# Rule 5: Remove exact duplicate rows
df_dml = df_dml.drop_duplicates().copy()

# Rule 6: Feature Engineering (Total Amount) & Standard DML Renaming
df_dml['total_amount'] = df_dml['quantity'] * df_dml['price']

df_dml = df_dml.rename(columns={
    'price': 'unit_price',
    'invoice': 'invoice_id',
    'stockcode': 'stock_code'
})

# Save DML Modeling-Ready Dataset
dml_output_path = OUTPUT_DIR / "cleaned_online_retail_dml.csv"
df_dml.to_csv(dml_output_path, index=False)

print(f"Final DML Modeling-Ready Shape: {df_dml.shape}")
print(f"Saved DML dataset to: {dml_output_path}")
print("--- Preprocessing Complete Successfully! ---")