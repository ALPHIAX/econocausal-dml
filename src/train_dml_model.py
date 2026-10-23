import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from econml.dml import LinearDML


INPUT_PATH = "Data/Processed/dml_modeling_data.csv"


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

df["invoicedate"] = pd.to_datetime(df["invoicedate"])


# --------------------------------------------------
# 2. Create temporal features
# --------------------------------------------------

df["year"] = df["invoicedate"].dt.year

df["month_sin"] = np.sin(
    2 * np.pi * df["invoicedate"].dt.month / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["invoicedate"].dt.month / 12
)

df["dow_sin"] = np.sin(
    2 * np.pi * df["invoicedate"].dt.dayofweek / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["invoicedate"].dt.dayofweek / 7
)

df["hour_sin"] = np.sin(
    2 * np.pi * df["invoicedate"].dt.hour / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["invoicedate"].dt.hour / 24
)


# --------------------------------------------------
# 3. Define treatment, outcome and X
# --------------------------------------------------

T = df["unit_price"].to_numpy()
Y = df["quantity"].to_numpy()

X = df[
    [
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


# --------------------------------------------------
# 4. Convert categorical variables
#    to pandas categorical dtype
# --------------------------------------------------

X["stock_code"] = X["stock_code"].astype("category").cat.codes
X["country"] = X["country"].astype("category").cat.codes


# --------------------------------------------------
# 5. Train/test split
# --------------------------------------------------

X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X,
    T,
    Y,
    test_size=0.20,
    random_state=42
)


print("=== DML Training Data ===")
print("Total observations:", len(X))
print("Total features:", X.shape[1])
print("Training observations:", len(X_train))
print("Test observations:", len(X_test))


# --------------------------------------------------
# 6. LightGBM nuisance models
# --------------------------------------------------

model_y = LGBMRegressor(
    
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model_t = LGBMRegressor(
    
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


# --------------------------------------------------
# 7. EconML Double Machine Learning
# --------------------------------------------------

dml = LinearDML(
    model_y=model_y,
    model_t=model_t,
    discrete_treatment=False,
    discrete_outcome=False,
    cv=2,
    random_state=42
)


# --------------------------------------------------
# 8. Train DML model
# --------------------------------------------------

print("\nTraining LinearDML with LightGBM...")

dml.fit(
    Y_train,
    T_train,
    X=X_train
)

print("DML training completed.")


# --------------------------------------------------
# 9. Estimate treatment effects
# --------------------------------------------------

effect = dml.effect(X_test)


# --------------------------------------------------
# 10. Treatment-effect summary
# --------------------------------------------------

print("\n=== Treatment Effect Results ===")
print("Number of effects:", len(effect))
print("Mean treatment effect:", np.mean(effect))
print("Median treatment effect:", np.median(effect))
print("Minimum treatment effect:", np.min(effect))
print("Maximum treatment effect:", np.max(effect))
print("Std treatment effect:", np.std(effect))

print("\n=== Treatment Effect Sanity Check ===")
print("Positive effects:", np.sum(effect > 0))
print("Negative effects:", np.sum(effect < 0))
print("Near-zero effects:", np.sum(np.isclose(effect, 0, atol=0.01)))

print("\n=== Effect Quantiles ===")
print(np.quantile(effect, [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]))

ite_output = pd.DataFrame({
    "estimated_ite": effect
})

ite_output.to_csv(
    "Data/Processed/dml_ite_results.csv",
    index=False
)

print("\nITE results saved to: Data/Processed/dml_ite_results.csv")
print("ITE rows:", len(ite_output))

print("\n=== ITE Distribution Validation ===")
print("Mean:", np.mean(effect))
print("Median:", np.median(effect))
print("Std:", np.std(effect))
print("5th percentile:", np.percentile(effect, 5))
print("95th percentile:", np.percentile(effect, 95))
print("IQR:", np.percentile(effect, 75) - np.percentile(effect, 25))

print("\nEffect range:")
print("Minimum:", np.min(effect))
print("Maximum:", np.max(effect))