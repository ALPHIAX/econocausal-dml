
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from econml.dml import NonParamDML
from sklearn.linear_model import Ridge


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
# 3. Select eligible non-POST products
# --------------------------------------------------

df["stock_code"] = df["stock_code"].astype(str).str.strip()

# Exclude postage and handling from merchandise analysis
df = df[
    df["stock_code"].str.upper() != "POST"
].copy()

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

print("\n========== SELECTED COHORT ==========")
print("Selected products:", df["stock_code"].nunique())
print("Selected observations:", len(df))
print("POST present:", (df["stock_code"].str.upper() == "POST").any())

if df["stock_code"].nunique() != 100:
    raise ValueError("Expected exactly 100 eligible products.")

if (df["stock_code"].str.upper() == "POST").any():
    raise ValueError("POST must not be included in the analysis.")

# --------------------------------------------------
# 4. Treatment and outcome
# --------------------------------------------------

T = df["unit_price"].to_numpy(dtype=np.float64)
Y = df["quantity"].to_numpy(dtype=np.float64)

dates = df["invoicedate"]

# --------------------------------------------------
# 5. Base temporal features
# --------------------------------------------------

year = dates.dt.year.to_numpy(dtype=np.float32)

# Center and scale year for numerical stability
year_mean = year.mean()
year_std = year.std()

if year_std == 0:
    raise ValueError("Year has no variation.")

year_scaled = (year - year_mean) / year_std

month_sin = np.sin(
    2 * np.pi * dates.dt.month.to_numpy() / 12
).astype(np.float32)

month_cos = np.cos(
    2 * np.pi * dates.dt.month.to_numpy() / 12
).astype(np.float32)

dow_sin = np.sin(
    2 * np.pi * dates.dt.dayofweek.to_numpy() / 7
).astype(np.float32)

dow_cos = np.cos(
    2 * np.pi * dates.dt.dayofweek.to_numpy() / 7
).astype(np.float32)

hour_sin = np.sin(
    2 * np.pi * dates.dt.hour.to_numpy() / 24
).astype(np.float32)

hour_cos = np.cos(
    2 * np.pi * dates.dt.hour.to_numpy() / 24
).astype(np.float32)

time_features = np.column_stack([
    year_scaled,
    month_sin,
    month_cos,
    dow_sin,
    dow_cos,
    hour_sin,
    hour_cos
]).astype(np.float32)

# --------------------------------------------------
# 6. Product encoding
# --------------------------------------------------

product_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=np.float32
)

product_features = product_encoder.fit_transform(
    df[["stock_code"]].astype(str)
)

# --------------------------------------------------
# 7. Construct final X matrix
# --------------------------------------------------

X = np.column_stack([
    time_features,
    product_features
]).astype(np.float32)

# --------------------------------------------------
# 9. Country adjustment variables (W)
# --------------------------------------------------

country_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=np.float32
)

W = country_encoder.fit_transform(
    df[["country"]].astype(str)
)

# --------------------------------------------------
# 10. Feature matrix validation
# --------------------------------------------------

if not np.isfinite(X).all():
    raise ValueError("Non-finite values detected in X.")

if not np.isfinite(W).all():
    raise ValueError("Non-finite values detected in W.")

if len(X) != len(df) or len(W) != len(df):
    raise ValueError("Feature matrix row alignment failed.")

print("\n========== FEATURE MATRIX ==========")
print("X shape:", X.shape)
print("W shape:", W.shape)
print("Product features:", product_features.shape[1])
print("Temporal features:", time_features.shape[1])
print("Total X features:", X.shape[1])
print("Estimated X memory (MB):", round(X.nbytes / (1024 ** 2), 2))


# --------------------------------------------------
# 5. Invoice-level train/test split
# --------------------------------------------------

groups = df["invoice_id"].astype(str).to_numpy()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, Y, groups=groups)
)

X_train, X_test = X[train_idx], X[test_idx]
W_train = W[train_idx]

T_train = T[train_idx]
Y_train = Y[train_idx]

groups_train = groups[train_idx]

train_invoices = set(groups[train_idx])
test_invoices = set(groups[test_idx])

assert train_invoices.isdisjoint(test_invoices)

print("Training observations:", len(train_idx))
print("Test observations:", len(test_idx))
print("Training invoices:", len(train_invoices))
print("Test invoices:", len(test_invoices))
print("Invoice overlap:", len(train_invoices & test_invoices))


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


dml = NonParamDML(
    model_y=model_y,
    model_t=model_t,
    model_final=Ridge(alpha=1.0),
    discrete_treatment=False,
    discrete_outcome=False,
    cv=GroupKFold(n_splits=2),
    random_state=RANDOM_STATE
)


# --------------------------------------------------
# 8. Fit
# --------------------------------------------------

print("\nTraining NonParamDML with Ridge final estimator...")

dml.fit(
    Y_train,
    T_train,
    X=X_train,
    W=W_train,
    groups=groups_train
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
