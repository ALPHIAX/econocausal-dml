
import numpy as np
import pandas as pd
from dowhy import CausalModel
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Data" / "Processed" / "cleaned_online_retail_dml.csv"

# --------------------------------------------------
# 1. Load and validate data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

required_columns = [
    "stock_code",
    "country",
    "invoicedate",
    "unit_price",
    "quantity",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df[required_columns].copy()

df["invoicedate"] = pd.to_datetime(
    df["invoicedate"],
    errors="coerce"
)

df = df.dropna(subset=required_columns)

df = df[
    (df["unit_price"] > 0) &
    (df["quantity"] > 0)
].copy()

# --------------------------------------------------
# 2. Match the DML model's 100-product cohort
# --------------------------------------------------

support = df.groupby("stock_code").agg(
    transactions=("unit_price", "size"),
    unique_prices=("unit_price", "nunique")
)

eligible_products = support[
    (support["transactions"] >= 100) &
    (support["unique_prices"] >= 2)
]

selected_products = (
    eligible_products
    .sort_values("transactions", ascending=False)
    .head(100)
    .index
)

df = df[df["stock_code"].isin(selected_products)].copy()

if df.empty:
    raise ValueError("No observations available for the selected products.")

print("Selected products:", df["stock_code"].nunique())
print("Selected observations:", len(df))

# --------------------------------------------------
# 3. Create temporal features
# --------------------------------------------------

df["year"] = df["invoicedate"].dt.year
df["month"] = df["invoicedate"].dt.month
df["dayofweek"] = df["invoicedate"].dt.dayofweek
df["hour"] = df["invoicedate"].dt.hour

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

time_features = [
    "year",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "hour_sin",
    "hour_cos",
]

# --------------------------------------------------
# 4. Encode categorical variables correctly
# --------------------------------------------------

product_dummies = pd.get_dummies(
    df["stock_code"].astype(str),
    prefix="product",
    drop_first=True,
    dtype=float
)

country_dummies = pd.get_dummies(
    df["country"].astype(str),
    prefix="country",
    drop_first=True,
    dtype=float
)

time_data = df[time_features].astype(float)

common_causes_data = pd.concat(
    [product_dummies, country_dummies, time_data],
    axis=1
)

data = pd.concat(
    [
        df[["unit_price", "quantity"]].astype(float),
        common_causes_data,
    ],
    axis=1
).replace([np.inf, -np.inf], np.nan).dropna()

common_causes = list(common_causes_data.columns)

print("\nAudit dataset shape:", data.shape)
print("Number of adjustment variables:", len(common_causes))

# --------------------------------------------------
# 5. Define and identify causal model
# --------------------------------------------------

causal_model = CausalModel(
    data=data,
    treatment="unit_price",
    outcome="quantity",
    common_causes=common_causes,
)

print("\nCausal Model:")
print(causal_model)

identified_estimand = causal_model.identify_effect(
    proceed_when_unidentifiable=True
)

print("\nIdentified Estimand:")
print(identified_estimand)

# --------------------------------------------------
# 6. Estimate causal effect
# --------------------------------------------------

estimate = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print("\nEstimated Effect:")
print(estimate)

# --------------------------------------------------
# 7. Placebo treatment refutation
# --------------------------------------------------

print("\nRunning placebo treatment refutation...")

placebo_result = causal_model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
    num_simulations=10,
    random_state=42,
)

print("\nPlacebo Refutation Result:")
print(placebo_result)

# --------------------------------------------------
# 8. Random common cause refutation
# --------------------------------------------------

print("\nRunning random common cause refutation...")

random_common_cause_result = causal_model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause",
    random_state=42,
)

print("\nRandom Common Cause Refutation Result:")
print(random_common_cause_result)

# --------------------------------------------------
# 9. Data subset refutation
# --------------------------------------------------

print("\nRunning data subset refutation...")

subset_result = causal_model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    random_state=42,
)

print("\nData Subset Refutation Result:")
print(subset_result)

# --------------------------------------------------
# 10. Audit summary
# --------------------------------------------------

print("\n=== Causal Audit Summary ===")
print(f"DoWhy estimated effect: {estimate.value:.6f}")

print("\nPlacebo refutation:")
print(placebo_result)

print("\nRandom common cause refutation:")
print(random_common_cause_result)

print("\nData subset refutation:")
print(subset_result)

print("\nConclusion:")
print(
    "The refutation tests provide diagnostic evidence about "
    "the stability of the estimated effect under the tested "
    "perturbations. They do not establish causal identification."
)

print(
    "The results must be interpreted alongside possible "
    "unobserved confounding, price endogeneity, overlap, "
    "and model specification limitations."
)
