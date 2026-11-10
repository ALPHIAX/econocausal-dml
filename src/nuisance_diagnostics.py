
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "Data" / "Processed" / "cleaned_online_retail_dml.csv"

RANDOM_STATE = 42
N_SPLITS = 2


# --------------------------------------------------
# 1. Reproduce the exact selected cohort
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required = [
    "invoice_id", "stock_code", "quantity",
    "invoicedate", "unit_price", "country"
]

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"], errors="coerce"
)
df["stock_code"] = df["stock_code"].astype(str).str.strip()

df = df.dropna(subset=required)
df = df[
    (df["unit_price"] > 0) &
    (df["quantity"] > 0) &
    (df["stock_code"].str.upper() != "POST")
].copy()

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

df = df[df["stock_code"].isin(selected)].copy()
df.reset_index(drop=True, inplace=True)

print("Selected cohort:", df.shape)


# --------------------------------------------------
# 2. Reproduce the training features
# --------------------------------------------------

dates = df["invoicedate"]

year = dates.dt.year.to_numpy(dtype=np.float32)
year_scaled = (year - year.mean()) / year.std()

month = dates.dt.month.to_numpy()
dow = dates.dt.dayofweek.to_numpy()
hour = dates.dt.hour.to_numpy()

month_sin = np.sin(2 * np.pi * month / 12).astype(np.float32)
month_cos = np.cos(2 * np.pi * month / 12).astype(np.float32)
dow_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32)
dow_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32)
hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)

temporal = np.column_stack([
    year_scaled,
    month_sin, month_cos,
    dow_sin, dow_cos,
    hour_sin, hour_cos
]).astype(np.float32)

product_codes = pd.get_dummies(
    df["stock_code"].astype(str),
    dtype=np.float32
).to_numpy()

product_year = product_codes * year_scaled[:, None]
product_month_sin = product_codes * month_sin[:, None]
product_month_cos = product_codes * month_cos[:, None]

X = np.column_stack([
    temporal,
    product_codes,
    product_year,
    product_month_sin,
    product_month_cos
]).astype(np.float32)

W = pd.get_dummies(
    df["country"].astype(str),
    dtype=np.float32
).to_numpy()

# Nuisance models receive X and W as covariates.
C = np.column_stack([X, W]).astype(np.float32)

T = df["unit_price"].to_numpy(dtype=np.float32)
Y = df["quantity"].to_numpy(dtype=np.float32)

print("X shape:", X.shape)
print("W shape:", W.shape)
print("Combined nuisance feature shape:", C.shape)


# --------------------------------------------------
# 3. Out-of-fold nuisance predictions
# --------------------------------------------------

oof_y = np.full(len(df), np.nan, dtype=np.float32)
oof_t = np.full(len(df), np.nan, dtype=np.float32)

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for fold, (train_idx, valid_idx) in enumerate(kf.split(C), start=1):

    print(f"\nFitting nuisance fold {fold}/{N_SPLITS}")

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

    model_y.fit(C[train_idx], Y[train_idx])
    model_t.fit(C[train_idx], T[train_idx])

    oof_y[valid_idx] = model_y.predict(C[valid_idx])
    oof_t[valid_idx] = model_t.predict(C[valid_idx])


# --------------------------------------------------
# 4. Evaluate out-of-fold predictions
# --------------------------------------------------

def report_metrics(name, actual, predicted):

    print(f"\n{name}")
    print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 5))
    print("MAE:", round(mean_absolute_error(actual, predicted), 5))
    print("R2:", round(r2_score(actual, predicted), 5))


if not np.isfinite(oof_y).all():
    raise ValueError("Invalid outcome predictions.")

if not np.isfinite(oof_t).all():
    raise ValueError("Invalid treatment predictions.")

report_metrics("OUTCOME MODEL: quantity", Y, oof_y)
report_metrics("TREATMENT MODEL: unit_price", T, oof_t)


# --------------------------------------------------
# 5. Residual diagnostics
# --------------------------------------------------

residual_y = Y - oof_y
residual_t = T - oof_t

print("\n========== OUTCOME RESIDUALS ==========")
print(pd.Series(residual_y).describe())

print("\n========== TREATMENT RESIDUALS ==========")
print(pd.Series(residual_t).describe())

print("\nResidual treatment standard deviation:",
      round(float(np.std(residual_t)), 6))

print("Residual treatment near-zero share:",
      round(float(np.mean(np.abs(residual_t) < 1e-6)), 6))

print("\n========== STAGE 7 VALIDATION ==========")

checks = {
    "All outcome predictions finite": np.isfinite(oof_y).all(),
    "All treatment predictions finite": np.isfinite(oof_t).all(),
    "All outcome residuals finite": np.isfinite(residual_y).all(),
    "All treatment residuals finite": np.isfinite(residual_t).all(),
    "Treatment residual variation positive": np.std(residual_t) > 0
}

for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

if not all(checks.values()):
    raise SystemExit("Nuisance diagnostic validation failed.")

print("\nSTAGE 7 DIAGNOSTICS COMPLETED")
