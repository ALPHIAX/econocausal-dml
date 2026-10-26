import pandas as pd

INPUT_PATH = "Data/Processed/dml_modeling_data.csv"
OUTPUT_PATH = "Data/Processed/dml_features.csv"


# 1. Load modeling dataset
df = pd.read_csv(INPUT_PATH)

# 2. Convert invoice date to datetime
df["invoicedate"] = pd.to_datetime(df["invoicedate"])

# 3. Create temporal features
df["year"] = df["invoicedate"].dt.year
df["month"] = df["invoicedate"].dt.month
df["day_of_week"] = df["invoicedate"].dt.dayofweek
df["hour"] = df["invoicedate"].dt.hour

# 4. Cyclical encoding for periodic time features
import numpy as np

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# 5. Keep only variables needed for the first DML pipeline
feature_df = df[
    [
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
        "hour_cos",
    ]
].copy()

# 6. Save feature-engineered dataset
feature_df.to_csv(OUTPUT_PATH, index=False)

print("=== DML Feature Engineering ===")
print("Shape:", feature_df.shape)
print("\nColumns:")
print(feature_df.columns.tolist())
print("\nMissing values:")
print(feature_df.isna().sum())
print("\nSaved:", OUTPUT_PATH)

