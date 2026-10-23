import pandas as pd
from dowhy import CausalModel

DATA_PATH = "Data/Processed/dml_modeling_data.csv"

df = pd.read_csv(DATA_PATH)

df["stock_code"] = df["stock_code"].astype("category").cat.codes
df["country"] = df["country"].astype("category").cat.codes

df["invoicedate"] = pd.to_datetime(df["invoicedate"])

df["year"] = df["invoicedate"].dt.year
df["month"] = df["invoicedate"].dt.month
df["dayofweek"] = df["invoicedate"].dt.dayofweek
df["hour"] = df["invoicedate"].dt.hour

df["month_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * df["month"] / 12)
df["month_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * df["month"] / 12)

df["dow_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * df["dayofweek"] / 7)
df["dow_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * df["dayofweek"] / 7)

df["hour_sin"] = __import__("numpy").sin(2 * __import__("numpy").pi * df["hour"] / 24)
df["hour_cos"] = __import__("numpy").cos(2 * __import__("numpy").pi * df["hour"] / 24)

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