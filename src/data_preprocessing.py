import pandas as pd
import numpy as np
import json
import os

print("--- Starting EconoCausal Data Preprocessing ---")

# 1. Load Raw JSON Datasets
portfolio = pd.read_json('Data/Raw/portfolio.json', orient='records', lines=True)
profile = pd.read_json('Data/Raw/profile.json', orient='records', lines=True)
transcript = pd.read_json('Data/Raw/transcript.json', orient='records', lines=True)

print("Raw datasets loaded successfully!")

# Helper function to convert unhashable types (lists, dicts) to strings for deduplication
def stringify_complex_cols(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
    return df

# 2. Clean Profile
profile.drop_duplicates(inplace=True)
profile['age'] = profile['age'].replace(118, np.nan)
profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d', errors='coerce')

# 3. Clean Portfolio
portfolio = stringify_complex_cols(portfolio)
portfolio.drop_duplicates(inplace=True)

# 4. Clean Transcript
transcript = stringify_complex_cols(transcript)
transcript.drop_duplicates(inplace=True)

# 5. Create Processed Directory
os.makedirs('Data/Processed', exist_ok=True)

# 6. Save Cleaned Datasets to CSV
profile.to_csv('Data/Processed/cleaned_profile.csv', index=False)
portfolio.to_csv('Data/Processed/cleaned_portfolio.csv', index=False)
transcript.to_csv('Data/Processed/cleaned_transcript.csv', index=False)

print("--- Preprocessing Complete: Cleaned files saved to Data/Processed/ ---")

# 5. Online Retail II Preprocessing
from pathlib import Path

print("\n--- Starting Online Retail II Preprocessing ---")

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "Data" / "Raw" / "online_retail_II.xlsx"
OUTPUT_DIR = ROOT / "Data" / "Processed"
OUTPUT_FILE = OUTPUT_DIR / "online_retail_cleaned.csv"

# Load all worksheets in the Excel workbook
sheets = pd.read_excel(RAW_FILE, sheet_name=None)

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

# Remove exact duplicate rows
retail = retail.drop_duplicates().copy()

# Convert key columns to appropriate types
retail["invoicedate"] = pd.to_datetime(
    retail["invoicedate"], errors="coerce"
)
retail["quantity"] = pd.to_numeric(
    retail["quantity"], errors="coerce"
)
retail["price"] = pd.to_numeric(
    retail["price"], errors="coerce"
)
retail["customer_id"] = pd.to_numeric(
    retail["customer_id"], errors="coerce"
)

# Remove rows missing essential transaction fields
retail = retail.dropna(
    subset=["invoice", "stockcode", "invoicedate", "quantity", "price"]
).copy()

# Preserve returns and flag records for investigation
retail["is_return"] = retail["quantity"] < 0
retail["customer_id_missing"] = retail["customer_id"].isna()
retail["transaction_value"] = retail["quantity"] * retail["price"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
retail.to_csv(OUTPUT_FILE, index=False)

print("Retail dataset shape:", retail.shape)
print("Saved to:", OUTPUT_FILE)
print("--- Online Retail II Preprocessing Complete ---")