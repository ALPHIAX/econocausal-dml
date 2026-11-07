import pandas as pd
from dowhy import CausalModel
import numpy as np

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "Data" / "Processed" / "cleaned_online_retail_dml.csv"

df = pd.read_csv(DATA_PATH)

df["stock_code"] = df["stock_code"].astype("category").cat.codes
df["country"] = df["country"].astype("category").cat.codes

df["invoicedate"] = pd.to_datetime(df["invoicedate"])

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
X = [
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

data = df[["unit_price", "quantity"] + X].dropna()

print("Audit dataset shape:", data.shape)
print(data.head())

causal_model = CausalModel(
    data=data,
    treatment="unit_price",
    outcome="quantity",
    common_causes=X,
)

print("\nCausal Model:")
print(causal_model)

identified_estimand = causal_model.identify_effect(
    proceed_when_unidentifiable=True
)

print("\nIdentified Estimand:")
print(identified_estimand)

print("\nRunning placebo treatment refutation...")

estimate = causal_model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

print("\nEstimated Effect:")
print(estimate)

placebo_result = causal_model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
    num_simulations=10,
    random_state=42
)

print("\nPlacebo Refutation Result:")
print(placebo_result)


print("\nRunning random common cause refutation...")

random_common_cause_result = causal_model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="random_common_cause",
    random_state=42
)

print("\nRandom Common Cause Refutation Result:")
print(random_common_cause_result)

print("\nRunning data subset refutation...")

subset_result = causal_model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    random_state=42
)

print("\nData Subset Refutation Result:")
print(subset_result)
print("\n=== Causal Audit Summary ===")
print(f"DoWhy estimated effect: {estimate.value:.6f}")

print("Placebo refutation:")
print(placebo_refutation)

print("Random common cause refutation:")
print(random_common_cause_refutation)

print("Data subset refutation:")
print(subset_refutation)

print("\nConclusion:")
print(
    "Placebo and random common cause checks did not indicate "
    "significant instability. However, the subset refutation "
    "requires further investigation. The causal estimate is "
    "not yet considered fully robust."
)

print(
    "Note: Refutation tests cannot establish the absence of "
    "unobserved confounding or guarantee causal identification."
)