
import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================
# 1. PROJECT PATHS
# =============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "Data" / "Processed" / "cleaned_online_retail_dml.csv"
OUTPUT_PATH = ROOT / "Data" / "Processed" / "dml_features.csv"


# =============================================================
# 2. LOAD CANONICAL DML DATASET
# =============================================================

print("\n=== DML Feature Engineering Started ===")

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Input dataset not found: {INPUT_PATH}"
    )

df = pd.read_csv(INPUT_PATH)

print("Input dataset shape:", df.shape)


# =============================================================
# 3. INPUT VALIDATION
# =============================================================

required_columns = [
    "unit_price",
    "quantity",
    "stock_code",
    "country",
    "invoicedate"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

if df.empty:
    raise ValueError("Input dataset is empty.")


# =============================================================
# 4. DATETIME CONVERSION
# =============================================================

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"],
    errors="coerce"
)

invalid_dates = df["invoicedate"].isna().sum()

if invalid_dates > 0:
    raise ValueError(
        f"Found {invalid_dates} invalid invoice dates."
    )


# =============================================================
# 5. TEMPORAL FEATURE ENGINEERING
# =============================================================

df["year"] = df["invoicedate"].dt.year
df["month"] = df["invoicedate"].dt.month
df["day_of_week"] = df["invoicedate"].dt.dayofweek
df["hour"] = df["invoicedate"].dt.hour


# =============================================================
# 6. CYCLICAL FEATURE ENCODING
# =============================================================

# Month: 12-month cycle
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

# Day of week: 7-day cycle
df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

# Hour: 24-hour cycle
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)


# =============================================================
# 7. SELECT DML FEATURES
# =============================================================

feature_columns = [
    "unit_price",
    "quantity",
    "stock_code",
    "country",
    "year",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "hour_sin",
    "hour_cos"
]

feature_df = df[feature_columns].copy()


# =============================================================
# 8. OUTPUT VALIDATION
# =============================================================

if feature_df.isnull().sum().sum() > 0:
    raise ValueError(
        "Missing values detected in engineered features."
    )

if not np.isfinite(
    feature_df.select_dtypes(include="number").to_numpy()
).all():
    raise ValueError(
        "Infinite values detected in numerical features."
    )

if len(feature_df) != len(df):
    raise ValueError(
        "Input and output row counts do not match."
    )


# =============================================================
# 9. SAVE FEATURE-ENGINEERED DATASET
# =============================================================

feature_df.to_csv(OUTPUT_PATH, index=False)

print("\n=== DML Feature Engineering Completed ===")

print("Input shape:", df.shape)
print("Output shape:", feature_df.shape)

print("\nOutput columns:")
print(feature_df.columns.tolist())

print("\nMissing values:")
print(feature_df.isnull().sum())

print("\nSaved:", OUTPUT_PATH)

