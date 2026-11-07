
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from econml.dml import SparseLinearDML


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "Data" / "Processed" / "cleaned_online_retail_dml.csv"
OUTPUT_PATH = ROOT / "Data" / "Processed" / "dml_ite_results.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20



# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required = [
    "invoice_id", "stock_code", "quantity",
    "invoicedate", "unit_price", "country"
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"], errors="coerce"
)

df = df.dropna(subset=required)

df = df[
    (df["unit_price"] > 0) &
    (df["quantity"] > 0)
].copy()

# --------------------------------------------------
# 3. Select products with observed price variation
# --------------------------------------------------

support = df.groupby("stock_code").agg(
    transactions=("unit_price", "size"),
    unique_prices=("unit_price", "nunique")
)

eligible = support[
    (support["transactions"] >= 100) &
    (support["unique_prices"] >= 2)
]

selected_products = (
    eligible.sort_values("transactions", ascending=False)
    .head(100)
    .index
)

df = df[df["stock_code"].isin(selected_products)].copy()
df = df.reset_index(drop=True)

print("Selected products:", df["stock_code"].nunique())
print("Selected observations:", len(df))

if df["stock_code"].nunique() < 2:
    raise ValueError("Insufficient eligible product categories.")

# --------------------------------------------------
# 4. Treatment, outcome and effect modifiers
# --------------------------------------------------

T = df["unit_price"].to_numpy(dtype=np.float64)
Y = df["quantity"].to_numpy(dtype=np.float64)

dates = df["invoicedate"]

time_features = pd.DataFrame({
    "year": dates.dt.year.astype(float),
    "month_sin": np.sin(2 * np.pi * dates.dt.month / 12),
    "month_cos": np.cos(2 * np.pi * dates.dt.month / 12),
    "dow_sin": np.sin(2 * np.pi * dates.dt.dayofweek / 7),
    "dow_cos": np.cos(2 * np.pi * dates.dt.dayofweek / 7),
    "hour_sin": np.sin(2 * np.pi * dates.dt.hour / 24),
    "hour_cos": np.cos(2 * np.pi * dates.dt.hour / 24),
})

product_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=np.float32
)

product_features = product_encoder.fit_transform(
    df[["stock_code"]].astype(str)
)

X = np.column_stack([
    time_features.to_numpy(dtype=np.float32),
    product_features
]).astype(np.float32)

# Country adjustment variables
country_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=np.float32
)

W = country_encoder.fit_transform(
    df[["country"]].astype(str)
)

# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42
)

X_train, X_test = X[train_idx], X[test_idx]
W_train = W[train_idx]

T_train = T[train_idx]
Y_train = Y[train_idx]

print("X dimensions:", X.shape)
print("W dimensions:", W.shape)
print("Training observations:", len(train_idx))
print("Test observations:", len(test_idx))


# --------------------------------------------------
# 6. Nuisance models
# --------------------------------------------------

model_y = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

model_t = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)


# --------------------------------------------------
# 7. Sparse Double Machine Learning
# --------------------------------------------------

dml = SparseLinearDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=False,
    discrete_outcome=False,
    cv=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# --------------------------------------------------
# 8. Fit
# --------------------------------------------------

print("\nTraining SparseLinearDML...")

dml.fit(
    Y_train,
    T_train,
    X=X_train,
    W=W_train
)

print("DML training completed.")


# --------------------------------------------------
# 9. Estimate conditional treatment effects
# --------------------------------------------------

effect = np.asarray(dml.effect(X_test)).reshape(-1)

if not np.isfinite(effect).all():
    raise ValueError("Non-finite treatment effects detected.")


# --------------------------------------------------
# 10. Save traceable CATE results
# --------------------------------------------------

test_metadata = df.iloc[test_idx]

ite_output = pd.DataFrame({
    "invoice_id": test_metadata["invoice_id"].to_numpy(),
    "stock_code": test_metadata["stock_code"].to_numpy(),
    "country": test_metadata["country"].to_numpy(),
    "unit_price": T[test_idx],
    "quantity": Y[test_idx],
    "estimated_cate": effect
})

ite_output.to_csv(OUTPUT_PATH, index=False)


# --------------------------------------------------
# 11. Summary
# --------------------------------------------------

print("\n=== CATE Summary ===")
print("Number of estimates:", len(effect))
print("Mean:", np.mean(effect))
print("Median:", np.median(effect))
print("Standard deviation:", np.std(effect))
print("Minimum:", np.min(effect))
print("Maximum:", np.max(effect))
print("Positive:", np.sum(effect > 0))
print("Negative:", np.sum(effect < 0))

print("\nSaved:", OUTPUT_PATH)
