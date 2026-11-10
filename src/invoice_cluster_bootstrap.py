
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

# --------------------------------------------------
# Invoice-cluster bootstrap pilot
# --------------------------------------------------

import time

N_BOOTSTRAPS = 200
BOOTSTRAP_SEED = 42

OUTPUT_PATH = ROOT / "Data" / "Processed" / "dml_product_cluster_bootstrap_replicates.csv"
PRODUCT_OUTPUT_PATH = ROOT / "Data" / "Processed" / "dml_product_cate_cluster_bootstrap.csv"
INDIVIDUAL_OUTPUT_PATH = ROOT / "Data" / "Processed" / "dml_individual_cate_cluster_bootstrap.csv"

rng = np.random.default_rng(BOOTSTRAP_SEED)

unique_invoices = np.unique(groups_train)

invoice_rows = {
    invoice: np.flatnonzero(groups_train == invoice)
    for invoice in unique_invoices
}

test_products = df.iloc[test_idx]["stock_code"].astype(str).to_numpy()

results = []
effect_draws = np.empty((N_BOOTSTRAPS, len(test_idx)), dtype=np.float64)

print("\n========== INVOICE CLUSTER BOOTSTRAP PILOT ==========")
print("Bootstrap repetitions:", N_BOOTSTRAPS)
print("Training invoice clusters:", len(unique_invoices))
print("Fixed test observations:", len(test_idx))

for b in range(N_BOOTSTRAPS):
    start_time = time.time()

    sampled_invoices = rng.choice(
        unique_invoices,
        size=len(unique_invoices),
        replace=True
    )

    row_chunks = [
        invoice_rows[invoice]
        for invoice in sampled_invoices
    ]

    bootstrap_indices = np.concatenate(row_chunks)

    # Keep every resampled copy of an original invoice in the same
    # cross-fitting group to prevent invoice leakage across folds.
    bootstrap_groups = groups_train[bootstrap_indices]

    X_boot = X_train[bootstrap_indices]
    W_boot = W_train[bootstrap_indices]
    T_boot = T_train[bootstrap_indices]
    Y_boot = Y_train[bootstrap_indices]

    model_y_boot = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    model_t_boot = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

    dml_boot = NonParamDML(
        model_y=model_y_boot,
        model_t=model_t_boot,
        model_final=Ridge(alpha=1.0),
        discrete_treatment=False,
        discrete_outcome=False,
        cv=GroupKFold(n_splits=2),
        random_state=RANDOM_STATE
    )

    dml_boot.fit(
        Y_boot,
        T_boot,
        X=X_boot,
        W=W_boot,
        groups=bootstrap_groups
    )

    effects = np.asarray(
        dml_boot.effect(X_test)
    ).reshape(-1)

    if not np.isfinite(effects).all():
        raise ValueError(
            f"Non-finite predictions in bootstrap repetition {b + 1}"
        )

    effect_draws[b, :] = effects

    product_frame = pd.DataFrame({
        "stock_code": test_products,
        "estimated_cate": effects
    })

    product_means = product_frame.groupby("stock_code")[
        "estimated_cate"
    ].mean()

    for product, mean_effect in product_means.items():
        results.append({
            "bootstrap_rep": b + 1,
            "stock_code": product,
            "mean_predicted_cate": mean_effect,
            "test_observations": int(
                (test_products == product).sum()
            )
        })

    elapsed = time.time() - start_time

    print(
        f"Bootstrap {b + 1}/{N_BOOTSTRAPS} completed "
        f"| rows={len(bootstrap_indices)} "
        f"| mean CATE={effects.mean():.4f} "
        f"| time={elapsed:.1f}s"
    )

replicate_df = pd.DataFrame(results)
replicate_df.to_csv(OUTPUT_PATH, index=False)

# Product-level bootstrap intervals
product_summary = replicate_df.groupby("stock_code").agg(
    bootstrap_mean_cate=("mean_predicted_cate", "mean"),
    bootstrap_se=("mean_predicted_cate", "std"),
    lower_95=("mean_predicted_cate", lambda s: s.quantile(0.025)),
    upper_95=("mean_predicted_cate", lambda s: s.quantile(0.975)),
    bootstrap_repetitions=("bootstrap_rep", "nunique"),
    test_observations=("test_observations", "first")
).reset_index()

product_summary.to_csv(PRODUCT_OUTPUT_PATH, index=False)

# Individual test-observation bootstrap intervals
test_metadata = df.iloc[test_idx].reset_index(drop=True)

baseline_path = ROOT / "Data" / "Processed" / "dml_ite_results.csv"
baseline = pd.read_csv(baseline_path)

if len(baseline) != len(test_metadata):
    raise ValueError("Baseline and test observation counts do not match.")

for column in ["invoice_id", "stock_code"]:
    left = baseline[column].astype(str).str.strip().to_numpy()
    right = test_metadata[column].astype(str).str.strip().to_numpy()

    if not np.array_equal(left, right):
        raise ValueError(f"Baseline observation alignment failed: {column}")

individual_summary = pd.DataFrame({
    "invoice_id": test_metadata["invoice_id"],
    "stock_code": test_metadata["stock_code"],
    "unit_price": T[test_idx],
    "quantity": Y[test_idx],
    "baseline_estimated_cate": baseline["estimated_cate"],
    "bootstrap_mean_cate": effect_draws.mean(axis=0),
    "bootstrap_se": effect_draws.std(axis=0, ddof=1),
    "lower_95": np.quantile(effect_draws, 0.025, axis=0),
    "upper_95": np.quantile(effect_draws, 0.975, axis=0)
})

individual_summary.to_csv(INDIVIDUAL_OUTPUT_PATH, index=False)

print("\n========== INVOICE CLUSTER BOOTSTRAP COMPLETE ==========")
print("Bootstrap repetitions:", N_BOOTSTRAPS)
print("Individual estimates:", len(individual_summary))
print("Product summaries:", len(product_summary))
print("Product bootstrap records:", len(replicate_df))
print("Individual output:", INDIVIDUAL_OUTPUT_PATH)
print("Product output:", PRODUCT_OUTPUT_PATH)
print("Replicate output:", OUTPUT_PATH)
print("Baseline alignment: PASS")
print("All bootstrap predictions finite: PASS")
print(
    "INTERPRETATION: Intervals measure training-sample bootstrap "
    "variation conditional on the fixed original test set. "
    "They do not establish causal identification."
)
